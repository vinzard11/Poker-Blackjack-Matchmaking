# Poker-Blackjack-Matchmaking-
A Machine learning model that matches players by assigning a skill level and putting them together in the same lobby for a rich game experience


I managed to get the most out of given data by creating 2 extra features which described a player's skill in a better way.

### POKER
#### 1.) PFRF_RATE:
* This parameter simply tells us if a player has high or low tendency to raise or fold in the beginning of the game.

* We can assume that smart players don’t raise unless and until they have great cards. 
So, for example if a player has PFRF below 10% we can assume he’s either got the 28 best combinations out of 1326 if he raises or he’s got bad cards that he doesn’t believe in. This is a kind of psychographic analysis and players with similar PFRF rates mean similar skills.

#### 2.) S_RAT:
* This rate tells us the frequency with which a player plays a hand at any given round in the game when he’s given an opportunity to bet or fold.
* This rate just says there is no reason to bet if you don’t have a good starting hand unless you want to bluff and if you do have a good starting hand it’s better to play aggressive by raising or betting.  And we already know playing aggressively is more profitable because you have two ways to win; having the best hand or causing your opponents to fold.

!(images/Poker_1.png)
* We can see the co





