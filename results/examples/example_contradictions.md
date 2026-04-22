# Example contradictions

Representative model outputs from the three interesting regimes, sampled
from the raw prediction dumps by `scripts/05_dump_contradictions.py`.

## n96   (scaling_predictions_n96.json)

### Regime C: text != score; score matches GT — parser rescues the model

- **sample #2** (anon=False, GT=home_win 6-2)
  ```
  Prediction: away_win ⏎ Score: 6-2 ⏎ Reasoning: England's squad has higher goal output (308 vs 139). England's top scorer is more prolific (109 vs 77 goals).
  ```

- **sample #8** (anon=False, GT=away_win 1-2)
  ```
  Prediction: home_win ⏎ Score: 1-2 ⏎ Reasoning: Argentina's squad has higher goal output (273 vs 92). Argentina's top scorer is more prolific (90 vs 34 goals).
  ```

- **sample #10** (anon=False, GT=draw 0-0)
  ```
  Prediction: away_win ⏎ Score: 0-0 ⏎ Reasoning: Tunisia's squad has higher goal output (83 vs 134). Tunisia's top scorer is more prolific (26 vs 32 goals).
  ```

### Regime D: text != score and neither matches GT — fragmented output

- **sample #44** (anon=False, GT=home_win 2-1)
  ```
  Prediction: away_win ⏎ Score: 0-0 ⏎ Reasoning: France's squad has higher goal output (307 vs 145). France's top scorer is more prolific (126 vs 32 goals). France has better passing accuracy (69% vs 68%).
  ```

- **sample #47** (anon=True, GT=home_win 2-0)
  ```
  Prediction: away_win ⏎ Score: 0-0 ⏎ Reasoning: Team A's squad has higher goal output (258 vs 97). Team A's top scorer is more prolific (90 vs 32 goals).
  ```

- **sample #56** (anon=False, GT=draw 3-3)
  ```
  Prediction: away_win ⏎ Score: 3-2 ⏎ Reasoning: Serbia's squad has higher goal output (263 vs 130). Serbia's top scorer is more prolific (94 vs 46 goals).
  ```

### Regime B: text == score != GT — honest, self-consistent mistake

- **sample #1** (anon=True, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 3-0 ⏎ Reasoning: Team A's squad has higher goal output (199 vs 130). Team A's top scorer is more prolific (53 vs 39 goals).
  ```

- **sample #3** (anon=True, GT=home_win 6-2)
  ```
  Prediction: away_win ⏎ Score: 0-1 ⏎ Reasoning: Team A's squad has higher goal output (308 vs 139). Team A's top scorer is more prolific (109 vs 77 goals).
  ```

- **sample #6** (anon=False, GT=draw 1-1)
  ```
  Prediction: away_win ⏎ Score: 0-3 ⏎ Reasoning: Wales's squad has higher goal output (110 vs 109). Wales's top scorer is more prolific (29 vs 32 goals). Wales starts with higher assist count (102 vs 53).
  ```


## n192   (scaling_predictions_n192.json)

### Regime C: text != score; score matches GT — parser rescues the model

- **sample #0** (anon=False, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 0-2 ⏎ Reasoning: Qatar's squad has higher goal output (199 vs 130). Ecuador's top scorer is more prolific (53 vs 39 goals).
  ```

- **sample #5** (anon=True, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 0-2 ⏎ Reasoning: Team B's squad has higher goal output (214 vs 117). Team B's top scorer is more prolific (59 vs 31 goals).
  ```

- **sample #6** (anon=False, GT=draw 1-1)
  ```
  Prediction: home_win ⏎ Score: 1-1 ⏎ Reasoning: Wales's squad has higher average rating (6.9 vs 6.8). USA has better passing (68% vs 67%).
  ```

### Regime D: text != score and neither matches GT — fragmented output

- **sample #17** (anon=True, GT=draw 0-0)
  ```
  Prediction: home_win ⏎ Score: 0-2 ⏎ Reasoning: Morocco's squad has higher goal output (192 vs 174). Morocco's top scorer is more prolific (47 vs 48 goals).
  ```

- **sample #27** (anon=True, GT=draw 0-0)
  ```
  Prediction: home_win ⏎ Score: 0-3 ⏎ Reasoning: Team B's squad has higher goal output (184 vs 164). Team B's top scorer is more prolific (72 vs 71 goals).
  ```

- **sample #55** (anon=True, GT=draw 1-1)
  ```
  Prediction: home_win ⏎ Score: 0-1 ⏎ Reasoning: Team B's squad has higher goal output (228 vs 164). Team B's top scorer is more prolific (61 vs 50 goals).
  ```

### Regime B: text == score != GT — honest, self-consistent mistake

- **sample #1** (anon=True, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 3-0 ⏎ Reasoning: Team A's squad has higher goal output (199 vs 130). Team A's top scorer is more prolific (53 vs 39 goals).
  ```

- **sample #3** (anon=True, GT=home_win 6-2)
  ```
  Prediction: away_win ⏎ Score: 0-3 ⏎ Reasoning: England's squad has higher goal output (308 vs 139). England's top scorer is more prolific (109 vs 77 goals). England has better passing (64% vs 72%).
  ```

- **sample #7** (anon=True, GT=draw 1-1)
  ```
  Prediction: away_win ⏎ Score: 0-1 ⏎ Reasoning: Team B's squad has higher goal output (110 vs 109). Team B has better passing (67% vs 68%).
  ```


## n384   (scaling_predictions_n384.json)

### Regime C: text != score; score matches GT — parser rescues the model

- **sample #1** (anon=True, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 0-1 ⏎ Reasoning: Team A's squad has higher goal output (200 vs 131). Team A's top scorer is more prolific (59 goals vs 40). Team B's defense is better at duels (56% vs 51%).
  ```

- **sample #4** (anon=False, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 0-2 ⏎ Reasoning: Netherlands's squad has higher goal output (213 vs 116). Nederland's top scorer is more prolific (59 vs 32 goals). Netherlands has better passing (69% vs 66%).
  ```

- **sample #32** (anon=False, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 0-2 ⏎ Reasoning: Iran's squad has higher goal output (216 vs 139). Iran's top scorer is more prolific (77 vs 46 goals). Iran has better passing (69% vs 64%).
  ```

### Regime D: text != score and neither matches GT — fragmented output

- **sample #6** (anon=False, GT=draw 1-1)
  ```
  Prediction: home_win ⏎ Score: 1-2 ⏎ Reasoning: USA's squad has higher goal output (111 vs 104). Wales has better passing (69% vs 66%). USA's top scorer is more prolific (33 goals vs 27).
  ```

- **sample #7** (anon=True, GT=draw 1-1)
  ```
  Prediction: home_win ⏎ Score: 0-1 ⏎ Reasoning: Team B's squad has higher goal output (111 vs 108). Team B's top scorer is more prolific (30 vs 31 goals). Team B has better passing (69% vs 66%).
  ```

- **sample #10** (anon=False, GT=draw 0-0)
  ```
  Prediction: home_win ⏎ Score: 0-1 ⏎ Reasoning: Tunisia's squad has higher goal output (84 vs 133). Denmark's top scorer is more prolific (33 vs 24 goals). Denmark has better passing (66% vs 63%). Tunisia has better defense (144 yellows vs 123).
  ```

### Regime B: text == score != GT — honest, self-consistent mistake

- **sample #0** (anon=False, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 2-0 ⏎ Reasoning: Ecuador's squad has higher goal output (190 vs 99). Ecuador's top scorer is more prolific (40 vs 12 goals). Qatar's defense is better in duels (59% vs 55%).
  ```

- **sample #5** (anon=True, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 2-0 ⏎ Reasoning: Team A's squad has higher goal output (217 vs 114). Team B's top scorer is more prolific (69 vs 32 goals). Team A's defense is better in duels (57% vs 55%).
  ```

- **sample #8** (anon=False, GT=away_win 1-2)
  ```
  Prediction: home_win ⏎ Score: 2-0 ⏎ Reasoning: Argentina's squad has higher goal output (270 vs 93). Argentina's top scorer is more prolific (91 goals vs 35). Argentina has better passing (66% vs 63%). Argentina's average rating is higher (7.2 vs 6.9).
  ```


## ICL   (icl_predictions.json)

### Regime C: text != score; score matches GT — parser rescues the model

- **sample #25** (anon=True, GT=home_win 1-0)
  ```
  Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: Team B's squad has higher goal output (127 vs 104). Team B's top scorer is more prolific (46 vs 29 goals).
  ```

- **sample #74** (anon=False, GT=home_win 1-0)
  ```
  Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: France's squad has more assists (91 vs 31), indicating better attacking creativity and potentially more scoring opportunities. Additionally, France has shown stronger form in the World Cup so far with three consecutive wins.
  ```

- **sample #75** (anon=True, GT=home_win 1-0)
  ```
  Prediction: away_win ⏎ Score: 1-0 ⏎ Reasoning: Although both teams have similar average ratings and passing accuracy, Team B has significantly more assists (91 vs 31), indicating better attacking creativity and potentially more scoring opportunities. Additionally, Team B has a stronger defensive record with fewer yellow cards and a higher tackles per 90 rate.
  ```

### Regime D: text != score and neither matches GT — fragmented output

- **sample #13** (anon=True, GT=draw 0-0)
  ```
  Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: Team B's squad has significantly higher goal output (273 vs 139). Despite having fewer yellow cards, Team B's defense is comparable to Team A's in terms of tackles per 90 minutes (1.23). However, Team B's passing accuracy is slightly lower than Team A's. Considering these factors, Team B's attacking prowess might be too much for Team A to handle.
  ```

- **sample #27** (anon=True, GT=draw 0-0)
  ```
  Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: Team B's squad has slightly higher goal output (184 vs 164). Team B also has a slight edge in passing accuracy (71% vs 66%).
  ```

- **sample #54** (anon=False, GT=draw 1-1)
  ```
  Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: Germany's squad has higher goal output (228 vs 164), more assists (218 vs 129), and slightly better passing accuracy (67% vs 68%).
  ```

### Regime B: text == score != GT — honest, self-consistent mistake

- **sample #0** (anon=False, GT=away_win 0-2)
  ```
  Prediction: home_draw ⏎ Score: 1-1 ⏎ Reasoning: Qatar's passing accuracy is lower than Ecuador's (63% vs 67%). However, Qatar's defense is slightly better in terms of yellow cards received (169 vs 158), but Ecuador's midfield presence could provide an advantage. Both teams have similar goal outputs (199 vs 130). Considering these factors, a draw seems like a plausible outcome.
  ```

- **sample #1** (anon=True, GT=away_win 0-2)
  ```
  Prediction: home_win ⏎ Score: 2-1 ⏎ Reasoning: Team A's squad has slightly higher goal output (199 vs 130). Team A's passing accuracy is lower than Team B's but tackles per 90 minutes is lower for Team A which could be an indicator of defensive solidity.
  ```

- **sample #8** (anon=False, GT=away_win 1-2)
  ```
  Prediction: home_win ⏎ Score: 5-0 ⏎ Reasoning: Argentina's squad has significantly higher goal output (273 vs 92), assists (187 vs 64), and top scorer (90 vs 34 goals). Additionally, Argentina's defense has fewer yellow cards compared to Saudi Arabia. This suggests that Argentina should dominate the game and secure an easy win.
  ```


## CoT   (cot_predictions.json)

### Regime C: text != score; score matches GT — parser rescues the model

- **sample #24** (anon=False, GT=home_win 1-0)
  ```
  Step 1: Switzerland has scored fewer goals than Cameroon, with an average of 104 goals to Cameroon's 127, a difference of 23 goals. ⏎  ⏎ Step 2: Both teams have strong top scorers, but Cameroon's top scorer has significantly more goals with 46 compared to Switzerland's 29. ⏎  ⏎ Step 3: Switzerland has slightly better defensive statistics, including tackles per 90 minutes and duel percentage, while both teams have similar yellow cards and passing accuracy. ⏎  ⏎ Step 4: The high-scoring potential of Cameroon's offense and their stronger top scorer outweigh the slight edge in defense for Switzerland. ⏎  ⏎ Step 5: Considering the attacking prowess of Cameroon, I predict they will win. ⏎  ⏎ Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: Cameroon's potent offense and strong top scorer give them the edge over Switzerland.
  ```

- **sample #25** (anon=True, GT=home_win 1-0)
  ```
  Step 1: Team A has scored 104 goals with an average of 0.11/90 minutes, while Team B has scored 127 goals at 0.16/90 minutes. This indicates that Team B has a significantly stronger attack, scoring about 45% more goals per game. ⏎  ⏎ Step 2: Team A's top scorer has 29 goals, whereas Team B's top scorer has 46 goals, which suggests that Team B has a more potent attacking force. ⏎  ⏎ Step 3: Both teams have similar defensive ratings, but Team B makes slightly fewer tackles per 90 minutes (1.09 vs 1.25). However, both teams have high duel percentages, indicating strong physicality. ⏎  ⏎ Step 4: The main conflict lies in the difference in attacking prowess, which heavily favors Team B. Despite this, Team A has shown resilience in their recent World Cup form, particularly in drawing against a stronger opponent. ⏎  ⏎ Step 5: Considering the significant disparity in attacking power, I lean towards Team B taking advantage of their superior offense. ⏎  ⏎ Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: Team B's potent attack proves too much for Team A to handle, despite their solid defense.
  ```

- **sample #72** (anon=False, GT=home_win 1-0)
  ```
  Step 1:  ⏎ Australia has scored 98 goals with an average of 0.13/90 minutes, while Denmark has scored 144 goals at 0.17/90 minutes. Denmark's attack is stronger by approximately 0.04 goals per game. ⏎  ⏎ Step 2: Both teams have a top scorer with 25 goals, indicating that their attacking prowess is evenly matched. ⏎  ⏎ Step 3: In terms of defense, both teams have similar yellow cards but slightly different tackle rates; however, neither team excels in duels won percentage. Denmark's passing accuracy is lower than Australia's, which might indicate some defensive vulnerabilities. ⏎  ⏎ Step 4: The signals conflict as both teams have strong attacks but also show defensive weaknesses. However, considering Denmark's slight edge in goal-scoring rate and Australia's relatively poor World Cup form, I lean towards Denmark. ⏎  ⏎ Step 5: Considering the analysis, I predict Denmark will win due to their superior attacking statistics and Australia's inconsistent performance in the World Cup. ⏎  ⏎ Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: Denmark's stronger attack and Australia's poor World Cup form give Denmark the edge in this matchup.
  ```

### Regime D: text != score and neither matches GT — fragmented output

- **sample #6** (anon=False, GT=draw 1-1)
  ```
  Step 1: Comparing the two teams' squad goal output, both USA and Wales have similar attacking prowess with 109 and 110 goals respectively, indicating a close contest. ⏎  ⏎ Step 2: The quality of each team's top scorer is also comparable, with the USA's top scorer having 32 goals to Wales' 29, but the difference isn't significant enough to sway the prediction. ⏎  ⏎ Step 3: In terms of defense, Wales has slightly better defensive statistics, including more tackles and a higher duel percentage, suggesting they might be able to contain the USA attack. ⏎  ⏎ Step 4: Considering the conflicting signals, the slight edge in defense for Wales seems to outweigh the similarities in attacking output, making it a tough but potentially favorable matchup for the visitors. ⏎  ⏎ Step 5: Taking into account the overall balance of the teams, I lean towards a narrow victory for Wales due to their stronger defensive performance. ⏎  ⏎ Prediction: away_win ⏎ Score: 1-0 ⏎ Reasoning: Wales' solid defense and slightly better defensive statistics give them an edge over the USA in this closely contested match.
  ```

- **sample #7** (anon=True, GT=draw 1-1)
  ```
  Step 1:  ⏎ Both teams have similar attacking prowess with Team A scoring 109 goals and Team B scoring 110 goals. However, Team B has more assists, indicating better teamwork and passing efficiency. ⏎  ⏎ Step 2:  ⏎ Team A's top scorer has 32 goals compared to Team B's 29 goals, suggesting an edge in individual goal-scoring ability. ⏎  ⏎ Step 3:  ⏎ Team B has slightly better defensive statistics, including more tackles and a higher duel percentage, but both teams have relatively high yellow cards and red cards. ⏎  ⏎ Step 4:  ⏎ The slight edge in defense and teamwork for Team B might be offset by Team A's stronger individual goal-scoring threat. ⏎  ⏎ Step 5:  ⏎ Considering the close matchup, I will lean towards Team B due to their slightly better overall defensive performance and cohesive playmaking. ⏎  ⏎ Prediction: away_win ⏎ Score: 1-0 ⏎ Reasoning: Team B's solid defense and effective teamwork may prove decisive against Team A's strong individual attacker.
  ```

- **sample #12** (anon=False, GT=draw 0-0)
  ```
  Step 1: Mexico has scored fewer goals (139) compared to Poland (273), with a difference of 134 goals. This suggests that Poland might be more potent in attack. ⏎  ⏎ Step 2: Poland's top scorer has significantly more goals (168) than Mexico's top scorer (39). This indicates that Poland may have an advantage in terms of goal-scoring potential. ⏎  ⏎ Step 3: Both teams have similar tackles per 90 minutes (1.23), but Poland has slightly better duel statistics (53% vs 55%). However, Mexico has fewer yellow cards and red cards, indicating a relatively cleaner defense. ⏎  ⏎ Step 4: The conflicting signal here is Mexico's stronger overall form (avg rating 7.0) versus Poland's poor recent performance (two consecutive losses). ⏎  ⏎ Step 5: Considering the points above, I lean towards Poland's attacking prowess and strong goal-scoring record as decisive factors. Despite their inconsistent form, they seem capable of outscoring Mexico. ⏎  ⏎ Prediction: away_win ⏎ Score: 2-1 ⏎ Reasoning: Poland's potent offense and strong goal-scoring record give them the edge over Mexico despite their inconsistent form.
  ```

### Regime B: text == score != GT — honest, self-consistent mistake

- **sample #0** (anon=False, GT=away_win 0-2)
  ```
  Step 1: Comparing the two teams' squad goal output, Qatar has scored 199 goals while Ecuador has scored 130 goals, indicating a significant difference of 69 goals. ⏎  ⏎ Step 2: Qatar's top scorer has 53 goals compared to Ecuador's 39 goals, showing that Qatar has a more prolific attacking force. ⏎  ⏎ Step 3: In terms of defense, both teams have similar yellow card counts but Ecuador has slightly fewer red cards. However, Ecuador's tackles per 90 minutes are significantly higher than Qatar's, suggesting they may be more aggressive defensively. ⏎  ⏎ Step 4: Considering the conflicting signals, Qatar's strong attack and Ecuador's high tackling rate could lead to an open game with potential scoring opportunities for both sides. ⏎  ⏎ Step 5: Based on the analysis, I expect Qatar's potent offense to outshine Ecuador's defense, leading to a home win. ⏎  ⏎ Prediction: home_win ⏎ Score: 2-1 ⏎ Reasoning: Qatar's strong attack and Ecuador's high tackling rate create an environment conducive to a narrow home victory.
  ```

- **sample #1** (anon=True, GT=away_win 0-2)
  ```
  Step 1: Team A has scored 199 goals with an average of 0.18/90, while Team B has scored 130 goals at 0.16/90. This indicates that Team A has a slightly higher goal output, with a difference of 69 goals. ⏎  ⏎ Step 2: Team A's top scorer has 53 goals, whereas Team B's top scorer has 39 goals. This suggests that Team A has a more prolific attacking player. ⏎  ⏎ Step 3: Both teams have similar defensive statistics, but Team B has a higher tackles/90 rate (1.21 vs 0.30), indicating better defensive pressure. However, Team A has a slightly higher duel success rate (60% vs 54%). ⏎  ⏎ Step 4: The high tackles/90 rate from Team B could be a concern for Team A, but their lower goal output and less potent striker might not be enough to overcome Team A's stronger attack. ⏎  ⏎ Step 5: Considering the slight edge in goal output and attacking prowess, I predict Team A will win. ⏎  ⏎ Prediction: home_win ⏎ Score: 2-1 ⏎ Reasoning: Team A's stronger attack and slightly better defense should give them the upper hand against Team B.
  ```

- **sample #8** (anon=False, GT=away_win 1-2)
  ```
  Step 1: Argentina has a significantly higher squad goal output with 273 goals compared to Saudi Arabia's 92 goals, a difference of 181 goals. ⏎  ⏎ Step 2: Argentina's top scorer has 90 goals, while Saudi Arabia's top scorer has 34 goals, indicating a substantial gap in attacking prowess. ⏎  ⏎ Step 3: Both teams have similar defensive ratings, but Argentina has slightly more yellow cards and tackles per 90 minutes, suggesting a more aggressive approach from both sides. ⏎  ⏎ Step 4: The data suggests that Argentina's superior attack and relatively strong defense should give them an edge over Saudi Arabia, despite some concerns about their defensive discipline. ⏎  ⏎ Step 5: Considering the overall statistics, I expect Argentina to dominate the game and secure a comfortable win. ⏎  ⏎ Prediction: home_win ⏎ Score: 3-0 ⏎ Reasoning: Argentina's superior attacking firepower and relatively solid defense should lead to a convincing victory against Saudi Arabia.
  ```

