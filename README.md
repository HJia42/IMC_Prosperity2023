# IMC_Prosperity2023

This is work from someone learning about algorithmic trading as well as working on improving CS/Coding skills as I am learning more or less on my own (Try not to use R/R Markdown). Currently top 50 working alone out of 1700+ active teams and 7000+ total teams who dropped out at one point or another. Overall, I would first like to thank IMC Trading for the oppertunity for not only running an incredible competition but also designing it in a way that challenges what seems to be multiple asepects of a quant trader while also doing so on a smooth and good learning curve starting with a simple task to more and more complicated tasks.

Leasons:

 - Slow down, try not to do too many things at once.
 - A good backtester is just important if not more than coming up with strategies
 - Keep note of specific changes you make to accomidate limitations of backtesters (this costed me around 70k + and probably top 20)
 - Understanding what might be Game Theory optimal may be nice but understanding your opponents is much more important. I was supprised by the amount of people who voted on the second manual trading round, wasn't as self selective as I thought and missed profitability range by 2-3 hundred points. (exected loss of around 50K)
 
 Strategies:
 
 Now most of the main ideas were more or less obvious for each, but 

- Pearls
  - Market making and arbitrage of 10k

- Bananas
  - Market making with a bit of directionality

- Coconuts/Pina Coladas
  - Pairs trading off of distribution of the difference and formulating a reward function to determine function of positional limits

- Berries
  - Time series based, hard coded most of this based on momentum and time thresholds

- Picnic/Dip/Ukulele/Baguette
  - Pairs Trading again but with more items where 3 are liquid but the last is very illiquid
  - decided to stay as neutral as possible, may have costed me more potential profit

- Gear/Dolphins
  - Event/News based
  - Deciphering what is and isn't a signal, which saved me from potential loss in Round 4

- Counter Party
  - Mainly tried finding potential positive strategies not market making (Caesar)
  - Maybe should have tried some sort of ML signal rf but lacked the time
  - Lots of "noice"/useless information and signal
  - Mainly traded on Olivia who was basically a prophet and knew the daily highs and lows.
  - Optimized Berries to include her info
  - Took a risk on the Pairs Trading on Picnic Baskets with the high and low signals on Ukulele
    - Since Picnic = 4 * Dips + 2 * Baguette + Ukulele + 360
    - We know that Ukulele = Picnic - 4 * Dips - 2 * Baguette - 360
    - Once given either signal go long one way and short the other
    - Depending on the chart for Round 5 this may hurt or benefit me.
