"""Task 4: greedy decoding, beam search, parsing to WikiSQL format, readable SQL.

Decoding never sees gold data: the only per-example information it uses is the
source token ids and the number of columns in the table (which the front end
also has, because the user types the column list).
"""
from __future__ import annotations

import re

import common
import torch
from common import AGG_OPS, BOS_ID, COND_OPS, EOS_ID, MAX_COLS, PAD_ID

MAX_DECODE_LEN = 64  # generated tokens, </s> included


# --------------------------------------------------------------- constraints
class ColumnConstraint:
    """Validity constraint applied to the next-token scores: <pad>, <s> and
    <sep> can never be generated, and neither can a column token <cK> with
    K >= number of columns in the table (such a query cannot exist). Everything
    else, including <unk>, is left to the model (<unk> is repaired by the parser)."""

    def __init__(self, sp, vocab_size: int):
        self.col_ids = torch.tensor([sp.piece_to_id(f"<c{i}>") for i in range(MAX_COLS)])
        self.always_banned = torch.tensor([PAD_ID, BOS_ID, sp.piece_to_id("<sep>")])
        self.vocab_size = vocab_size

    def banned_mask(self, n_cols: torch.Tensor) -> torch.Tensor:
        """n_cols: (B,) -> bool (B, V); True = banned."""
        b = n_cols.size(0)
        mask = torch.zeros(b, self.vocab_size, dtype=torch.bool, device=n_cols.device)
        mask[:, self.always_banned.to(n_cols.device)] = True
        too_far = torch.arange(MAX_COLS, device=n_cols.device)[None, :] >= n_cols[:, None]
        mask[:, self.col_ids.to(n_cols.device)] = too_far
        return mask


# ------------------------------------------------------------------- greedy
@torch.no_grad()
def greedy_decode(model, src, n_cols=None, constraint=None, max_len=MAX_DECODE_LEN):
    """Batched greedy decoding. src: (B, S) -> list of B token-id lists (no <s>/</s>)."""
    model.eval()
    device = src.device
    memory, src_mask = model.encode(src)
    b = src.size(0)
    ys = torch.full((b, 1), BOS_ID, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    banned = constraint.banned_mask(n_cols.to(device)) if constraint is not None else None
    for _ in range(max_len):
        logits = model.decode(ys, memory, src_mask)[:, -1].float()  # (B, V)
        if banned is not None:
            logits = logits.masked_fill(banned, float("-inf"))
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        done |= nxt == EOS_ID
        if done.all():
            break
    return [_strip(row.tolist()) for row in ys[:, 1:]]


# --------------------------------------------------------------------- beam
def _length_penalty(length: int, alpha: float) -> float:
    """GNMT length penalty, used with alpha = 0.6 in the Transformer paper (Sec. 6.1)."""
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(model, src, n_cols=None, constraint=None, beam_size=4,
                max_len=MAX_DECODE_LEN, alpha=0.6):
    """Beam search for ONE example. src: (1, S) -> best token-id list (no <s>/</s>)."""
    model.eval()
    device = src.device
    memory, src_mask = model.encode(src)
    banned = None
    if constraint is not None:
        banned = constraint.banned_mask(n_cols.view(1).to(device))  # (1, V)

    alive = [([BOS_ID], 0.0)]  # (tokens, sum log-prob)
    finished = []              # (tokens, normalised score)
    for step in range(max_len):
        k = len(alive)
        ys = torch.tensor([t for t, _ in alive], dtype=torch.long, device=device)
        logp = torch.log_softmax(
            model.decode(ys, memory.expand(k, -1, -1), src_mask.expand(k, -1, -1, -1))[:, -1].float(), -1)
        if banned is not None:
            logp = logp.masked_fill(banned, float("-inf"))
        if step == max_len - 1:  # length cap reached: force </s>
            forced = torch.full_like(logp, float("-inf"))
            forced[:, EOS_ID] = logp[:, EOS_ID]
            logp = forced
        scores = logp + torch.tensor([s for _, s in alive], device=device)[:, None]
        top_s, top_i = scores.view(-1).topk(min(2 * beam_size, scores.numel()))
        vocab = logp.size(-1)
        new_alive = []
        for s, i in zip(top_s.tolist(), top_i.tolist()):
            if s == float("-inf"):
                break
            seq = alive[i // vocab][0] + [i % vocab]
            if i % vocab == EOS_ID:
                finished.append((seq, s / _length_penalty(len(seq) - 1, alpha)))
            else:
                new_alive.append((seq, s))
            if len(new_alive) == beam_size:
                break
        alive = new_alive
        if not alive:
            break
        # stop once the best finished hypothesis cannot be beaten: scores only
        # decrease, and the most optimistic penalty for an alive beam is at max_len
        if len(finished) >= beam_size:
            best_alive = max(s for _, s in alive) / _length_penalty(max_len, alpha)
            if max(f for _, f in finished) >= best_alive:
                break
    if not finished:  # should not happen because of the forced </s>, kept for safety
        finished = [(t, s / _length_penalty(len(t), alpha)) for t, s in alive]
    best = max(finished, key=lambda x: x[1])[0]
    return _strip(best[1:])


def _strip(ids):
    out = []
    for t in ids:
        if t in (EOS_ID, PAD_ID):
            break
        out.append(t)
    return out


# ------------------------------------------------------------------ parsing
_SQL_RE = re.compile(r"select\s+(?:(max|min|count|sum|avg)\s+)?<c(\d+)>(?:\s+where\s+(.+))?")
_COND_RE = re.compile(r"<c(\d+)>\s*(=|>|<)\s*(.+)")
_COND_SPLIT = re.compile(r"\s+and\s+(?=<c\d+>)")  # a value may contain the word "and"
_AGG_LOWER = [a.lower() for a in AGG_OPS]


UNK_SURFACE = "\u2047"  # what sp.decode prints for <unk>


def repair_unk(value: str, question: str | None) -> str:
    """Characters never seen in training (e.g. Hebrew or CJK names) are encoded
    as <unk> and decode to '\u2047'. Values are almost always copied from the
    question, so recover the original characters by matching the known pieces
    around each '\u2047' against the (lower-cased) question. Uses only the model
    input, never gold data. Returns the value unchanged if there is no match."""
    if UNK_SURFACE not in value or not question:
        return value
    parts = [re.escape(p.strip()) for p in value.split(UNK_SURFACE)]
    pattern = r"\s*\S{1,8}?\s*".join(parts)
    m = re.search(pattern, question.lower())
    return m.group(0).strip() if m else value


def parse_to_wikisql(text: str, n_cols: int, question: str | None = None):
    """'select count <c3> where <c1> = kim manners' -> {'sel','agg','conds'} or None.
    `question` (optional) enables <unk> repair of condition values."""
    m = _SQL_RE.fullmatch(text.strip())
    if m is None:
        return None
    agg = _AGG_LOWER.index(m.group(1)) if m.group(1) else 0
    sel = int(m.group(2))
    if sel >= n_cols:
        return None
    conds = []
    if m.group(3):
        for part in _COND_SPLIT.split(m.group(3).strip()):
            c = _COND_RE.fullmatch(part.strip())
            if c is None:
                return None
            col, op = int(c.group(1)), COND_OPS.index(c.group(2))
            val = repair_unk(c.group(3).strip(), question)
            if col >= n_cols or not val:
                return None
            conds.append([col, op, val])
    return {"sel": sel, "agg": agg, "conds": conds}


def _is_number(v) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except ValueError:
        return False


def to_readable_sql(query: dict, header: list[str]) -> str:
    """WikiSQL dict -> SQL with the real column names."""
    def col(i):
        name = header[i] if i < len(header) else f"col{i}"
        return '"' + name.replace('"', '""') + '"'

    def lit(v):
        return str(v) if _is_number(v) else "'" + str(v).replace("'", "''") + "'"

    sel = col(query["sel"])
    if query["agg"]:
        sel = f"{AGG_OPS[query['agg']]}({sel})"
    sql = f"SELECT {sel} FROM table"
    if query["conds"]:
        sql += " WHERE " + " AND ".join(
            f"{col(c)} {COND_OPS[o]} {lit(v)}" for c, o, v in query["conds"])
    return sql


# --------------------------------------------------------- convenience API
class SQLGenerator:
    """Question + column names -> parsed query + readable SQL. Used by the front end."""

    def __init__(self, ckpt=common.BEST_CKPT, device=None, constrained=True):
        self.device = device or common.get_device()
        self.model, self.ckpt = common.load_model(ckpt, self.device)
        self.sp = common.load_sp()
        self.constraint = ColumnConstraint(self.sp, self.sp.get_piece_size()) if constrained else None

    def __call__(self, question: str, header: list[str], method: str = "beam"):
        src_text = common.data_prep.encode_source(question, header)
        src = torch.tensor([self.sp.encode(src_text) + [EOS_ID]], device=self.device)
        n_cols = torch.tensor([len(header)], device=self.device)
        if method == "beam":
            ids = beam_search(self.model, src, n_cols, self.constraint)
        else:
            ids = greedy_decode(self.model, src, n_cols, self.constraint)[0]
        raw = self.sp.decode(ids)
        query = parse_to_wikisql(raw, len(header), question)
        readable = to_readable_sql(query, header) if query else None
        return {"raw": raw, "query": query, "sql": readable}
