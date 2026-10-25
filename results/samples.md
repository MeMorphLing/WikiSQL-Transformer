# Qualitative samples (dev, beam decoding)

Correct = logical form and execution both match the gold query. Values are lower-cased by the preprocessing, so the generated SQL is lower-case.

## Correct

### Correct 1 (dev #5108)

**Question:** What date was the opponent the Boston Yanks?  
**Columns:** Week, Date, Opponent, Result, Record  
**Gold:** `SELECT "Date" FROM table WHERE "Opponent" = 'boston yanks'`  
**Ours:** `SELECT "Date" FROM table WHERE "Opponent" = 'boston yanks'`  

### Correct 2 (dev #5703)

**Question:** What 1st leg has Alense Vivaldi (Trentino) as Agg.?  
**Columns:** Team 1, Agg., Team 2, 1st leg, 2nd leg  
**Gold:** `SELECT "1st leg" FROM table WHERE "Agg." = 'alense vivaldi (trentino)'`  
**Ours:** `SELECT "1st leg" FROM table WHERE "Agg." = 'alense vivaldi (trentino)'`  

### Correct 3 (dev #980)

**Question:** If the parallel bars is 16.150, who is the gymnast?  
**Columns:** Position, Gymnast, Floor, Pommel Horse, Rings, Vault, Parallel Bars, Horizontal Bar, Total  
**Gold:** `SELECT "Gymnast" FROM table WHERE "Parallel Bars" = 16.150`  
**Ours:** `SELECT "Gymnast" FROM table WHERE "Parallel Bars" = 16.150`  

### Correct 4 (dev #7360)

**Question:** What is the average Gold entry for the Netherlands that also has a Bronze entry that is greater than 0?  
**Columns:** Nation, Gold, Silver, Bronze, Total  
**Gold:** `SELECT AVG("Gold") FROM table WHERE "Nation" = 'netherlands' AND "Bronze" > 0`  
**Ours:** `SELECT AVG("Gold") FROM table WHERE "Nation" = 'netherlands' AND "Bronze" > 0`  

### Correct 5 (dev #8307)

**Question:** Name the rank with finish of 12 and year of 1963  
**Columns:** Year, Start, Qual, Rank, Finish, Laps  
**Gold:** `SELECT "Rank" FROM table WHERE "Finish" = 12 AND "Year" = 1963`  
**Ours:** `SELECT "Rank" FROM table WHERE "Finish" = 12 AND "Year" = 1963`  

## Wrong

### Wrong 1 (dev #1036)

**Question:** What is the location that has the 12th district?  
**Columns:** District, Location, Representative, Cook PVI, 2012 election results, Median household income (2011)  
**Gold:** `SELECT "Location" FROM table WHERE "District" = '12th'`  
**Ours:** `SELECT "Representative" FROM table WHERE "District" = '12th'`  
**Failure:** wrong column  

### Wrong 2 (dev #7897)

**Question:** Name the position for indiana state  
**Columns:** Season, Player, School, Position, Class  
**Gold:** `SELECT "Position" FROM table WHERE "School" = 'indiana state'`  
**Ours:** `SELECT "Position" FROM table WHERE "Class" = 'indiana state'`  
**Failure:** wrong condition column  

### Wrong 3 (dev #6619)

**Question:** What is the Venue of the Competition on 2001-05-05?  
**Columns:** Date, Venue, Score, Result, Competition  
**Gold:** `SELECT "Venue" FROM table WHERE "Date" = '2001-05-05'`  
**Ours:** `SELECT "Venue" FROM table WHERE "Date" = '2001-05-05-05-05'`  
**Failure:** wrong value  

### Wrong 4 (dev #3576)

**Question:** What is the week of the game played on November 28, 1974?  
**Columns:** Week, Date, Opponent, Result, Attendance  
**Gold:** `SELECT MIN("Week") FROM table WHERE "Date" = 'november 28, 1974'`  
**Ours:** `SELECT COUNT("Week") FROM table WHERE "Date" = 'november 28, 1974'`  
**Failure:** wrong aggregation  

### Wrong 5 (dev #783)

**Question:** When dizzee rascal is 5th, who was the winner?  
**Columns:** Poll Year, Winner, Second, Third, Fourth, Fifth, Sixth, Seventh, Eighth, Ninth, Tenth  
**Gold:** `SELECT "Winner" FROM table WHERE "Fifth" = 'Dizzee Rascal'`  
**Ours:** `SELECT "Winner" FROM table WHERE "Tenth" = '5th' AND "Second" = 'dizzee rascal'`  
**Failure:** extra condition  
