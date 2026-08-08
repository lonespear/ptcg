# Top-5 team profiles — how they climbed, what they play

Built 2026-08-07. Ratings/ranks from the 2026-08-06 leaderboard snapshot; per-episode histories from Meta Kaggle (through 2026-08-06 13:40 UTC); submission identity recovered by peeking TeamNames in one replay per submission; decks from full replays sampled across each submission's lifetime. Raw numbers: `top5_profile.json`.

## Synthesis

**The portfolio question: answered, negative.** Across ~380 sampled replays
covering 40 submissions of the five teams, every submission played exactly one
fixed 60-card list — zero within-submission deck variation anywhere. Deck
variety lives ACROSS submissions: teams use Kaggle's two active-submission
slots as champion + challenger and resubmit to change lists.

**Cross-team patterns, in numbers:**

1. **Rating is submission-scoped.** Every new submission starts at Elo 600.
   Over the 33 mapped top-5 submissions that peaked >=1100, the median climb
   was **9 episodes to 1000** and **16 episodes to 1100**. The climb is cheap
   (~half a day of ladder time); the plateau winrate is the product. Team
   winrates at altitude: flg 60.5%, James & Henry 61.0%, Raihan 59.4%,
   @kdcyberdude 58.2%, LiamK 57.2% (all mapped episodes).
2. **Iteration cadence.** LiamK: 14 submissions in 31 days (8 different
   archetypes). flg: 8 in 11 days (7 archetypes). James & Henry: 8 in 14 days
   (one archetype, tech-tweaked). Raihan: 7, with a 20-day absence
   (Jul 11 - Aug 1). @kdcyberdude: 3 in 4 days. Four of five currently run
   one proven list + one experiment; Raihan instead runs the same exact list
   in both slots.
3. **The Grimmsnarl netdeck.** A byte-identical 60-card Marnie's Grimmsnarl
   ex / Munkidori / Froslass list (signature 7:10,104:2,112:4,646:4,... in
   `history_decklists.csv`) is played by **143 agents** on the ladder at a
   field winrate of 46.5% (1,247/2,684 deck-games on 08-06). Four of the five
   top teams have fielded it (Raihan and @kdcyberdude hold top-5 with the
   STOCK list at 58-59% winrate; flg's current copy is a ~3-card retune).
   The edge at the top is the agent, not list secrecy.
4. **Entry date does not gate rank.** @kdcyberdude's first mapped episode is
   Aug 2; top-5 within 2 days. flg entered Jul 26 and hit #1 in ~11 days.
   LiamK has grinded since at least Jul 6 and sits #2.
5. **Archetypes at the top right now:** Grimmsnarl/Munkidori (flg slot 1,
   Raihan both slots, @kdcyberdude slot 1), Mega Kangaskhan multi-attacker
   toolbox (James & Henry, both slots), Mega Lopunny/Mega Froslass (LiamK's
   1185-rated score-setter), plus experiments (Hydrapple, Mega Lucario,
   Kangaskhan wall). Mono Teal Mask Ogerpon — our archetype — appears in the
   top five only as LiamK's Jul 6 submission: peak 1110, retired Jul 9.

**Tactical lessons for Lemmes Yad (target 1000+):**

1. **Champion/challenger, fast cadence.** A submission proves itself inside
   ~30 episodes. Pin the best agent+list in slot 1; iterate slot 2 every 1-2
   days. Judge an experiment by winrate after episode ~30, not by its rating
   mid-climb (600 -> 1000 costs a good agent ~9-13 games; the reset is
   noise).
2. **List quality is table stakes and free.** Two top-5 teams run the public
   143-agent Grimmsnarl list unmodified; our own mined data carries its exact
   signature. Sustained 1000+ requires ~55%+ winrate against 1000-1100
   opposition; a proven engine list + our search agent is the shortest path,
   and mono-Ogerpon's known loss modes cap us below that bar.
3. **Netdeck first, tune later.** flg reached #1 on stock/near-stock lists of
   whatever archetype was winning (Grimmsnarl netdeck day 1, then Mewtwo,
   then a reused Dragapult list, then a retuned Grimmsnarl). None of the five
   invented a new archetype; they out-piloted the field on known lists.

**Coverage caveats.** Submission identity comes from a replay-peek sweep of
every submission that ever peaked >=1100 (cut short at ~57% done) plus local
mined-day joins, so: LiamK/Raihan submissions before Jul 6 that never peaked
1100 would be missed; @kdcyberdude shows ladder presence Jul 30 - Aug 2 with
no mapped submission (likely 1-2 subs peaking <1100); James & Henry deck
samples cover their Aug submissions (July subs are Kangaskhan-era by
archetype continuity, not replay-verified). Meta Kaggle data ends 2026-08-06
13:40 UTC.


## 1. flg — 1164.5

8 submissions mapped, 3722 episodes, 2250W-1471L (60.5%). First mapped episode 2026-07-26 15:06.

| sub | first ep | last ep | n | W-L | ep to 1000 | ep to 1100 | peak | final | best W streak |
|---|---|---|---|---|---|---|---|---|---|
| 55004495 | 07-26 15:06 | 07-27 15:45 | 392 | 261-131 | 7 | 9 | 1255.5 | 1228.3 | 14 |
| 55004691 | 07-26 15:17 | 07-27 16:13 | 421 | 271-150 | 10 | 16 | 1222.3 | 1169.0 | 9 |
| 55033309 | 07-27 16:06 | 08-05 14:48 | 1028 | 585-443 | 6 | 14 | 1227.5 | 1100.2 | 13 |
| 55033457 | 07-27 16:13 | 08-05 16:02 | 1512 | 876-635 | 10 | 12 | 1219.3 | 1035.5 | 13 |
| 55275219 | 08-05 16:08 | 08-06 05:45 | 147 | 105-42 | 14 | 25 | 1212.1 | 1204.1 | 10 |
| 55275287 | 08-05 16:11 | 08-06 06:17 | 97 | 59-38 | 11 | 18 | 1136.8 | 1099.1 | 10 |
| 55290684 | 08-06 06:24 | 08-06 12:49 | 66 | 47-19 | 13 | 51 | 1130.9 | 1122.7 | 8 |
| 55290732 | 08-06 06:26 | 08-06 12:45 | 59 | 46-13 | 46 | - | 1028.8 | 1011.4 | 20 |

**Deck repertoire** (97 replays sampled, 7 distinct exact lists):

- **Hydrapple ex** — 27 sampled games, subs [55290732]
  - 13x Basic {G} Energy (1), 2x Applin (92), 2x Dipplin (93), 4x Teal Mask Ogerpon ex (96), 1x Fezandipiti ex (140), 2x Hydrapple ex (150), 2x Bayleef (709), 2x Meganium (710), 2x Chikorita (917), 2x Tapu Bulu (920), 2x Meowth ex (1071), 1x Unfair Stamp (1080), 4x Bug Catching Set (1094), 2x Night Stretcher (1097), 4x Ultra Ball (1121), 3x Poké Pad (1152), 2x Boss’s Orders (1182), 1x Lana’s Aid (1184), 4x Lillie's Determination (1227), 1x Dawn (1231), 4x Forest of Vitality (1261)
- **Marnie's Grimmsnarl ex** — 18 sampled games, subs [55290684]
  - 10x Basic {D} Energy (7), 2x Froslass (104), 4x Munkidori (112), 4x Marnie's Impidimp (646), 3x Marnie's Morgrem (647), 3x Marnie's Grimmsnarl ex (648), 2x Snorunt (860), 3x Rare Candy (1079), 1x Unfair Stamp (1080), 4x Buddy-Buddy Poffin (1086), 3x Night Stretcher (1097), 1x Pokégear 3.0 (1122), 4x Poké Pad (1152), 2x Handheld Fan (1161), 2x Boss’s Orders (1182), 4x Team Rocket's Petrel (1219), 4x Lillie's Determination (1227), 4x Spikemuth Gym (1259)
- **Dragapult ex** — 16 sampled games, subs [55004495, 55275219]
  - 4x Basic {R} Energy (2), 4x Basic {P} Energy (5), 2x Basic {D} Energy (7), 2x Munkidori (112), 4x Dreepy (119), 4x Drakloak (120), 3x Dragapult ex (121), 1x Fezandipiti ex (140), 2x Budew (235), 1x Meowth ex (1071), 1x Unfair Stamp (1080), 4x Buddy-Buddy Poffin (1086), 2x Night Stretcher (1097), 4x Crushing Hammer (1120), 4x Ultra Ball (1121), 4x Poké Pad (1152), 3x Boss’s Orders (1182), 3x Crispin (1198), 1x Judge (1213), 4x Lillie's Determination (1227), 1x Dawn (1231), 2x Jamming Tower (1246)
- **Marnie's Grimmsnarl ex** — 10 sampled games, subs [55004691]
  - 10x Basic {D} Energy (7), 2x Froslass (104), 4x Munkidori (112), 4x Marnie's Impidimp (646), 3x Marnie's Morgrem (647), 3x Marnie's Grimmsnarl ex (648), 2x Snorunt (860), 3x Rare Candy (1079), 1x Unfair Stamp (1080), 4x Buddy-Buddy Poffin (1086), 3x Night Stretcher (1097), 1x Pokégear 3.0 (1122), 1x Tool Scrapper (1137), 4x Poké Pad (1152), 2x Boss’s Orders (1182), 4x Team Rocket's Petrel (1219), 4x Lillie's Determination (1227), 1x Dawn (1231), 4x Spikemuth Gym (1259)
- **Team Rocket's Mewtwo ex** — 10 sampled games, subs [55033309]
  - 7x Basic {G} Energy (1), 4x Team Rocket's Energy (15), 4x Team Rocket's Tarountula (400), 4x Team Rocket's Spidops (401), 2x Team Rocket's Articuno (414), 2x Team Rocket's Mewtwo ex (431), 3x Team Rocket's Mimikyu (434), 2x Buddy-Buddy Poffin (1086), 3x Bug Catching Set (1094), 1x Ultra Ball (1121), 4x Team Rocket's Transceiver (1134), 4x Poké Pad (1152), 1x Hero’s Cape (1159), 1x Brave Bangle (1175), 4x Team Rocket's Ariana (1216), 1x Team Rocket's Archer (1217), 3x Team Rocket's Giovanni (1218), 4x Team Rocket's Proton (1220), 3x Lillie's Determination (1227), 3x Team Rocket's Factory (1257)
- **Mega Kangaskhan ex** — 10 sampled games, subs [55033457]
  - 2x Basic {G} Energy (1), 4x Mist Energy (11), 4x Spiky Energy (14), 4x Grow Grass Energy (18), 2x Rock Fighting Energy (20), 1x Cornerstone Mask Ogerpon ex (117), 4x Dwebble (344), 3x Crustle (345), 2x Mega Kangaskhan ex (756), 2x Buddy-Buddy Poffin (1086), 2x Ultra Ball (1121), 4x Pokégear 3.0 (1122), 1x Switch (1123), 1x Tool Scrapper (1137), 4x Jumbo Ice Cream (1147), 1x Hero’s Cape (1159), 4x Boss’s Orders (1182), 2x Colress’s Tenacity (1194), 1x Xerosic’s Machinations (1197), 4x Team Rocket's Petrel (1219), 2x Hilda (1225), 4x Lillie's Determination (1227), 1x Team Rocket's Factory (1257), 1x Battle Cage (1264)
- **Fezandipiti ex** — 6 sampled games, subs [55275287]
  - 3x Basic {P} Energy (5), 4x Telepath Psychic Energy (19), 3x Dudunsparce (66), 1x Fezandipiti ex (140), 3x Dunsparce (305), 4x Abra (741), 4x Kadabra (742), 4x Alakazam (743), 3x Rare Candy (1079), 4x Enhanced Hammer (1081), 4x Buddy-Buddy Poffin (1086), 1x Night Stretcher (1097), 1x Sacred Ash (1129), 4x Poké Pad (1152), 1x Lucky Helmet (1156), 3x Boss’s Orders (1182), 1x Lana’s Aid (1184), 3x Xerosic’s Machinations (1197), 3x Hilda (1225), 4x Dawn (1231), 1x Neutralization Zone (1247), 1x Nighttime Mine (1266)

No within-submission list variation observed in the sample: one exact 60-card list per submission.

**Matchups (sampled games, W-L by opponent archetype):**

| opponent archetype | W | L |
|---|---|---|
| Marnie's Grimmsnarl ex | 35 | 6 |
| Fezandipiti ex | 4 | 6 |
| Mega Lucario ex | 4 | 4 |
| Mega Kangaskhan ex | 5 | 2 |
| Team Rocket's Mewtwo ex | 7 | 0 |
| Mega Lopunny ex | 4 | 1 |
| Dragapult ex | 3 | 1 |
| Cynthia's Garchomp ex | 2 | 2 |
| Teal Mask Ogerpon ex | 4 | 0 |
| Hydrapple ex | 2 | 0 |
| Crustle | 2 | 0 |
| Mega Abomasnow ex | 1 | 0 |
| Dudunsparce | 1 | 0 |
| Cornerstone Mask Ogerpon ex | 1 | 0 |


## 2. LiamK — 1158.9

14 submissions mapped, 11476 episodes, 6564W-4905L (57.2%). First mapped episode 2026-07-06 18:17.

| sub | first ep | last ep | n | W-L | ep to 1000 | ep to 1100 | peak | final | best W streak |
|---|---|---|---|---|---|---|---|---|---|
| 54403820 | 07-06 18:17 | 07-09 06:25 | 868 | 505-362 | 18 | 155 | 1110.5 | 1056.6 | 8 |
| 54405730 | 07-06 20:00 | 07-09 18:05 | 1048 | 640-408 | 6 | 193 | 1154.1 | 1059.9 | 12 |
| 54484720 | 07-09 06:26 | 07-13 05:19 | 649 | 381-268 | 109 | 282 | 1160.0 | 1069.1 | 10 |
| 54503495 | 07-09 18:27 | 07-16 06:51 | 1856 | 1023-833 | 9 | 11 | 1217.3 | 1039.1 | 12 |
| 54634530 | 07-13 05:36 | 07-20 01:05 | 1941 | 1068-873 | 7 | 26 | 1177.2 | 1022.0 | 10 |
| 54754477 | 07-16 06:55 | 07-20 01:13 | 799 | 422-377 | 6 | 9 | 1192.9 | 1006.1 | 9 |
| 54895414 | 07-22 05:58 | 07-26 06:59 | 1001 | 564-435 | 7 | 19 | 1178.1 | 1040.8 | 13 |
| 54907538 | 07-22 16:11 | 07-26 21:40 | 870 | 516-352 | 11 | 153 | 1173.4 | 1093.3 | 12 |
| 55011514 | 07-26 21:41 | 08-01 04:46 | 1028 | 606-421 | 8 | 11 | 1212.8 | 1093.8 | 11 |
| 55090635 | 07-29 19:15 | 08-02 22:03 | 760 | 436-324 | 7 | 16 | 1211.1 | 1107.0 | 15 |
| 55195669 | 08-02 22:03 | 08-04 19:44 | 199 | 109-89 | 9 | 12 | 1194.1 | 1103.2 | 8 |
| 55195673 | 08-02 22:03 | 08-04 19:32 | 114 | 76-38 | 71 | - | 1024.4 | 1008.9 | 17 |
| 55248957 | 08-04 19:58 | 08-06 12:33 | 116 | 79-37 | 64 | - | 1049.7 | 1035.2 | 8 |
| 55248965 | 08-04 19:59 | 08-06 13:25 | 227 | 139-88 | 8 | 13 | 1195.6 | 1185.5 | 12 |

**Deck repertoire** (144 replays sampled, 10 distinct exact lists):

- **Fezandipiti ex** — 30 sampled games, subs [54503495, 54634530, 54754477]
  - 3x Basic {P} Energy (5), 4x Telepath Psychic Energy (19), 3x Dudunsparce (66), 1x Fezandipiti ex (140), 4x Dunsparce (305), 4x Abra (741), 4x Kadabra (742), 4x Alakazam (743), 4x Rare Candy (1079), 4x Enhanced Hammer (1081), 4x Buddy-Buddy Poffin (1086), 1x Night Stretcher (1097), 1x Sacred Ash (1129), 4x Poké Pad (1152), 3x Boss’s Orders (1182), 1x Lana’s Aid (1184), 3x Xerosic’s Machinations (1197), 3x Hilda (1225), 4x Dawn (1231), 1x Neutralization Zone (1247)
- **Marnie's Grimmsnarl ex** — 20 sampled games, subs [55011514, 55090635]
  - 10x Basic {D} Energy (7), 2x Froslass (104), 4x Munkidori (112), 4x Marnie's Impidimp (646), 3x Marnie's Morgrem (647), 3x Marnie's Grimmsnarl ex (648), 2x Snorunt (860), 3x Rare Candy (1079), 1x Unfair Stamp (1080), 4x Buddy-Buddy Poffin (1086), 3x Night Stretcher (1097), 1x Pokégear 3.0 (1122), 1x Tool Scrapper (1137), 4x Poké Pad (1152), 2x Boss’s Orders (1182), 4x Team Rocket's Petrel (1219), 4x Lillie's Determination (1227), 1x Dawn (1231), 4x Spikemuth Gym (1259)
- **Cornerstone Mask Ogerpon ex** — 17 sampled games, subs [54405730, 54484720]
  - 3x Basic {G} Energy (1), 1x Basic {W} Energy (3), 1x Basic {F} Energy (6), 3x Basic {D} Energy (7), 4x Prism Energy (16), 2x Grow Grass Energy (18), 4x Munkidori (112), 2x Cornerstone Mask Ogerpon ex (117), 4x Dwebble (344), 4x Crustle (345), 2x Team Rocket's Articuno (414), 4x Buddy-Buddy Poffin (1086), 1x Crushing Hammer (1120), 4x Jumbo Ice Cream (1147), 4x Poké Pad (1152), 1x Hero’s Cape (1159), 4x Crispin (1198), 4x Brock’s Scouting (1210), 4x Lillie's Determination (1227), 4x Urbain (1236)
- **Mega Lucario ex** — 14 sampled games, subs [55248957]
  - 13x Basic {F} Energy (6), 2x Makuhita (673), 2x Hariyama (674), 2x Lunatone (675), 3x Solrock (676), 3x Riolu (677), 4x Mega Lucario ex (678), 4x Ultra Ball (1121), 2x Switch (1123), 4x Premium Power Pro (1141), 4x Fighting Gong (1142), 4x Poké Pad (1152), 1x Hero’s Cape (1159), 2x Boss’s Orders (1182), 4x Judge (1213), 4x Lillie's Determination (1227), 2x Wally's Compassion (1229)
- **Mega Lopunny ex** — 14 sampled games, subs [55248965]
  - 3x Basic {W} Energy (3), 4x Mist Energy (11), 1x Enriching Energy (13), 3x Dudunsparce (66), 1x Fan Rotom (174), 4x Dunsparce (305), 2x Buneary (848), 2x Mega Lopunny ex (849), 2x Snorunt (860), 2x Mega Froslass ex (861), 4x Buddy-Buddy Poffin (1086), 3x Hand Trimmer (1087), 4x Ultra Ball (1121), 2x Pokégear 3.0 (1122), 4x Poké Pad (1152), 3x Air Balloon (1174), 2x Boss’s Orders (1182), 3x Hilda (1225), 4x Lillie's Determination (1227), 4x Wally's Compassion (1229), 3x Battle Cage (1264)
- **Mega Kangaskhan ex** — 10 sampled games, subs [54895414]
  - 1x Basic {G} Energy (1), 4x Mist Energy (11), 4x Spiky Energy (14), 4x Grow Grass Energy (18), 4x Dwebble (344), 4x Crustle (345), 4x Mega Kangaskhan ex (756), 4x Buddy-Buddy Poffin (1086), 2x Hand Trimmer (1087), 4x Pokégear 3.0 (1122), 4x Switch (1123), 4x Jumbo Ice Cream (1147), 1x Hero’s Cape (1159), 2x Boss’s Orders (1182), 4x Xerosic’s Machinations (1197), 4x Hilda (1225), 4x Lillie's Determination (1227), 2x Battle Cage (1264)
- **Team Rocket's Mewtwo ex** — 10 sampled games, subs [54907538]
  - 9x Basic {G} Energy (1), 4x Team Rocket's Energy (15), 4x Team Rocket's Tarountula (400), 4x Team Rocket's Spidops (401), 2x Team Rocket's Articuno (414), 2x Team Rocket's Mewtwo ex (431), 3x Team Rocket's Mimikyu (434), 3x Bug Catching Set (1094), 1x Ultra Ball (1121), 4x Team Rocket's Transceiver (1134), 4x Poké Pad (1152), 1x Hero’s Cape (1159), 1x Brave Bangle (1175), 4x Team Rocket's Ariana (1216), 1x Team Rocket's Archer (1217), 3x Team Rocket's Giovanni (1218), 4x Team Rocket's Proton (1220), 3x Lillie's Determination (1227), 3x Team Rocket's Factory (1257)
- **Mega Kangaskhan ex** — 8 sampled games, subs [55195669]
  - 4x Mist Energy (11), 4x Spiky Energy (14), 4x Grow Grass Energy (18), 2x Rock Fighting Energy (20), 1x Cornerstone Mask Ogerpon ex (117), 4x Dwebble (344), 3x Crustle (345), 2x Mega Kangaskhan ex (756), 2x Buddy-Buddy Poffin (1086), 4x Crushing Hammer (1120), 2x Ultra Ball (1121), 4x Pokégear 3.0 (1122), 1x Switch (1123), 4x Jumbo Ice Cream (1147), 1x Hero’s Cape (1159), 4x Boss’s Orders (1182), 1x Xerosic’s Machinations (1197), 1x Cook (1212), 4x Team Rocket's Petrel (1219), 3x Hilda (1225), 4x Lillie's Determination (1227), 1x Team Rocket's Factory (1257)
- **Mega Starmie ex** — 8 sampled games, subs [55195673]
  - 9x Basic {W} Energy (3), 2x Froslass (104), 4x Snorunt (860), 2x Mega Froslass ex (861), 3x Staryu (1030), 3x Mega Starmie ex (1031), 4x Buddy-Buddy Poffin (1086), 2x Hand Trimmer (1087), 1x Night Stretcher (1097), 3x Pokégear 3.0 (1122), 3x Mega Signal (1145), 2x Poké Pad (1152), 1x Hero’s Cape (1159), 2x Boss’s Orders (1182), 1x Salvatore (1189), 2x Xerosic’s Machinations (1197), 2x Cheren (1224), 2x Hilda (1225), 4x Lillie's Determination (1227), 4x Wally's Compassion (1229), 4x Surfing Beach (1262)
- **Teal Mask Ogerpon ex** — 5 sampled games, subs [54403820]
  - 13x Basic {G} Energy (1), 4x Mist Energy (11), 4x Spiky Energy (14), 4x Grow Grass Energy (18), 2x Teal Mask Ogerpon ex (96), 4x Dwebble (344), 4x Crustle (345), 4x Buddy-Buddy Poffin (1086), 1x Night Stretcher (1097), 4x Jumbo Ice Cream (1147), 1x Hero’s Cape (1159), 4x Xerosic’s Machinations (1197), 4x Cook (1212), 4x Lillie's Determination (1227), 3x Waitress (1235)

No within-submission list variation observed in the sample: one exact 60-card list per submission.

**Matchups (sampled games, W-L by opponent archetype):**

| opponent archetype | W | L |
|---|---|---|
| Marnie's Grimmsnarl ex | 22 | 16 |
| Fezandipiti ex | 16 | 12 |
| Mega Kangaskhan ex | 8 | 4 |
| Mega Lucario ex | 6 | 3 |
| Mega Lopunny ex | 5 | 3 |
| Dragapult ex | 3 | 3 |
| Team Rocket's Mewtwo ex | 4 | 2 |
| Cynthia's Garchomp ex | 3 | 2 |
| Alakazam | 3 | 1 |
| Mega Starmie ex | 2 | 2 |
| Archaludon ex | 2 | 0 |
| Cornerstone Mask Ogerpon ex | 0 | 2 |
| Bloodmoon Ursaluna | 0 | 1 |
| Hop’s Snorlax | 1 | 0 |
| Team Rocket's Honchkrow | 0 | 1 |
| Dudunsparce | 0 | 1 |
| Brambleghast | 1 | 0 |
| Mega Abomasnow ex | 1 | 0 |
| Flareon ex | 1 | 0 |
| Team Rocket's Articuno | 1 | 0 |
| Teal Mask Ogerpon ex | 0 | 1 |
| Iono’s Bellibolt ex | 0 | 1 |
| Arboliva ex | 1 | 0 |
| Ethan's Typhlosion | 1 | 0 |


## 3. Raihan Ramadistra — 1139.0

7 submissions mapped, 3059 episodes, 1816W-1243L (59.4%). First mapped episode 2026-07-07 08:03.

| sub | first ep | last ep | n | W-L | ep to 1000 | ep to 1100 | peak | final | best W streak |
|---|---|---|---|---|---|---|---|---|---|
| 54421283 | 07-07 08:03 | 07-08 01:05 | 192 | 114-78 | 11 | - | 1087.4 | 1019.2 | 7 |
| 54445400 | 07-08 01:13 | 07-09 15:09 | 603 | 366-237 | 17 | 137 | 1145.1 | 1070.8 | 10 |
| 54499492 | 07-09 15:43 | 07-11 05:15 | 642 | 381-261 | 7 | 10 | 1172.5 | 1112.7 | 10 |
| 55153037 | 08-01 03:42 | 08-02 04:23 | 112 | 65-47 | 13 | - | 1054.6 | 1031.9 | 8 |
| 55171940 | 08-01 22:35 | 08-03 05:07 | 328 | 185-143 | 11 | 15 | 1237.8 | 1116.6 | 11 |
| 55177269 | 08-02 04:23 | 08-06 13:29 | 800 | 466-334 | 13 | 66 | 1226.2 | 1116.1 | 9 |
| 55202823 | 08-03 05:09 | 08-06 12:37 | 382 | 239-143 | 34 | 88 | 1182.0 | 1127.3 | 11 |

**Deck repertoire** (72 replays sampled, 3 distinct exact lists):

- **Marnie's Grimmsnarl ex** — 44 sampled games, subs [55153037, 55171940, 55177269, 55202823]
  - 10x Basic {D} Energy (7), 2x Froslass (104), 4x Munkidori (112), 4x Marnie's Impidimp (646), 3x Marnie's Morgrem (647), 3x Marnie's Grimmsnarl ex (648), 2x Snorunt (860), 3x Rare Candy (1079), 1x Unfair Stamp (1080), 4x Buddy-Buddy Poffin (1086), 3x Night Stretcher (1097), 1x Pokégear 3.0 (1122), 1x Tool Scrapper (1137), 4x Poké Pad (1152), 2x Boss’s Orders (1182), 4x Team Rocket's Petrel (1219), 4x Lillie's Determination (1227), 1x Dawn (1231), 4x Spikemuth Gym (1259)
- **Mega Kangaskhan ex** — 20 sampled games, subs [54445400, 54499492]
  - 1x Basic {G} Energy (1), 4x Mist Energy (11), 4x Spiky Energy (14), 4x Grow Grass Energy (18), 3x Dwebble (344), 3x Crustle (345), 4x Mega Kangaskhan ex (756), 3x Buddy-Buddy Poffin (1086), 1x Hand Trimmer (1087), 1x Ultra Ball (1121), 3x Pokégear 3.0 (1122), 2x Switch (1123), 4x Jumbo Ice Cream (1147), 1x Hero’s Cape (1159), 1x Handheld Fan (1161), 4x Boss’s Orders (1182), 2x Eri (1186), 1x Xerosic’s Machinations (1197), 1x Lisia’s Appeal (1204), 4x Team Rocket's Petrel (1219), 3x Hilda (1225), 4x Lillie's Determination (1227), 1x Community Center (1242), 1x Team Rocket's Factory (1257)
- **Marnie's Grimmsnarl ex** — 8 sampled games, subs [54421283]
  - 10x Basic {D} Energy (7), 3x Dudunsparce (66), 3x Munkidori (112), 1x Fezandipiti ex (140), 1x Budew (235), 3x Dunsparce (305), 4x Marnie's Impidimp (646), 2x Marnie's Morgrem (647), 4x Marnie's Grimmsnarl ex (648), 1x Yveltal (689), 4x Rare Candy (1079), 4x Buddy-Buddy Poffin (1086), 1x Tool Scrapper (1137), 4x Poké Pad (1152), 1x Hero’s Cape (1159), 2x Boss’s Orders (1182), 1x Xerosic’s Machinations (1197), 4x Lillie's Determination (1227), 3x Dawn (1231), 3x Spikemuth Gym (1259), 1x Risky Ruins (1260)

No within-submission list variation observed in the sample: one exact 60-card list per submission.

**Matchups (sampled games, W-L by opponent archetype):**

| opponent archetype | W | L |
|---|---|---|
| Marnie's Grimmsnarl ex | 11 | 8 |
| Fezandipiti ex | 9 | 5 |
| Mega Kangaskhan ex | 6 | 3 |
| Mega Lopunny ex | 2 | 2 |
| Cynthia's Garchomp ex | 3 | 1 |
| Dudunsparce | 1 | 3 |
| Alakazam | 2 | 2 |
| Teal Mask Ogerpon ex | 1 | 2 |
| Cornerstone Mask Ogerpon ex | 2 | 0 |
| Mega Lucario ex | 1 | 1 |
| Archaludon ex | 2 | 0 |
| Dragapult ex | 1 | 0 |
| Mega Venusaur ex | 0 | 1 |
| Hydrapple ex | 0 | 1 |
| Brambleghast | 1 | 0 |
| Mega Starmie ex | 1 | 0 |


## 4. James Cox & Henry Chao — 1138.6

8 submissions mapped, 3762 episodes, 2295W-1467L (61.0%). First mapped episode 2026-07-23 20:03.

| sub | first ep | last ep | n | W-L | ep to 1000 | ep to 1100 | peak | final | best W streak |
|---|---|---|---|---|---|---|---|---|---|
| 54935503 | 07-23 20:03 | 07-28 18:33 | 1045 | 639-406 | 38 | 191 | 1252.2 | 1141.7 | 10 |
| 54954310 | 07-24 15:18 | 07-30 13:30 | 1139 | 683-456 | 7 | 9 | 1202.9 | 1159.5 | 14 |
| 55063047 | 07-28 19:05 | 07-31 10:32 | 455 | 277-178 | 7 | 11 | 1217.0 | 1166.1 | 14 |
| 55135508 | 07-31 11:11 | 08-01 21:34 | 344 | 214-130 | 28 | 70 | 1177.1 | 1134.4 | 11 |
| 55188658 | 08-02 14:58 | 08-05 17:56 | 393 | 239-154 | 20 | 95 | 1172.5 | 1139.6 | 8 |
| 55188672 | 08-02 14:59 | 08-05 17:56 | 154 | 94-60 | 31 | - | 1037.1 | 1027.3 | 9 |
| 55278235 | 08-05 18:31 | 08-06 13:17 | 106 | 67-39 | 9 | 14 | 1213.3 | 1108.7 | 14 |
| 55278240 | 08-05 18:31 | 08-06 12:09 | 126 | 82-44 | 6 | 10 | 1140.9 | 1124.5 | 10 |

**Deck repertoire** (147 replays sampled, 2 distinct exact lists):

- **Mega Kangaskhan ex** — 78 sampled games, subs [54935503, 54954310, 55063047, 55135508, 55188658, 55188672]
  - 9x Basic {G} Energy (1), 2x Basic {W} Energy (3), 2x Basic {L} Energy (4), 1x Basic {P} Energy (5), 2x Basic {F} Energy (6), 2x Raging Bolt ex (63), 3x Teal Mask Ogerpon ex (96), 1x Wellspring Mask Ogerpon ex (108), 1x Fezandipiti ex (140), 2x Latias ex (184), 1x Lillie’s Clefairy ex (272), 3x Mega Kangaskhan ex (756), 1x Passimian (978), 3x Meowth ex (1071), 1x Prime Catcher (1088), 2x Night Stretcher (1097), 2x Glass Trumpet (1098), 4x Energy Switch (1116), 4x Ultra Ball (1121), 2x Boss’s Orders (1182), 2x Xerosic’s Machinations (1197), 4x Crispin (1198), 2x Cyrano (1205), 4x Area Zero Underdepths (1250)
- **Mega Kangaskhan ex** — 39 sampled games, subs [55278235, 55278240]
  - 9x Basic {G} Energy (1), 1x Basic {W} Energy (3), 2x Basic {L} Energy (4), 2x Basic {P} Energy (5), 2x Basic {F} Energy (6), 2x Raging Bolt ex (63), 3x Teal Mask Ogerpon ex (96), 1x Wellspring Mask Ogerpon ex (108), 1x Fezandipiti ex (140), 2x Latias ex (184), 1x Lillie’s Clefairy ex (272), 3x Mega Kangaskhan ex (756), 1x Passimian (978), 3x Meowth ex (1071), 1x Prime Catcher (1088), 2x Night Stretcher (1097), 2x Glass Trumpet (1098), 4x Energy Switch (1116), 4x Ultra Ball (1121), 2x Boss’s Orders (1182), 2x Xerosic’s Machinations (1197), 4x Crispin (1198), 2x Cyrano (1205), 4x Area Zero Underdepths (1250)

No within-submission list variation observed in the sample: one exact 60-card list per submission.

**Matchups (sampled games, W-L by opponent archetype):**

| opponent archetype | W | L |
|---|---|---|
| Marnie's Grimmsnarl ex | 33 | 11 |
| Mega Kangaskhan ex | 8 | 16 |
| Fezandipiti ex | 15 | 4 |
| Mega Lopunny ex | 3 | 5 |
| Cynthia's Garchomp ex | 2 | 4 |
| Teal Mask Ogerpon ex | 0 | 6 |
| Dragapult ex | 2 | 2 |
| Ceruledge ex | 2 | 0 |
| Crustle | 2 | 0 |
| Mega Lucario ex | 2 | 0 |


## 5. @kdcyberdude — 1135.8

3 submissions mapped, 661 episodes, 385W-276L (58.2%). First mapped episode 2026-08-02 10:19.

| sub | first ep | last ep | n | W-L | ep to 1000 | ep to 1100 | peak | final | best W streak |
|---|---|---|---|---|---|---|---|---|---|
| 55183271 | 08-02 10:19 | 08-03 16:39 | 205 | 123-82 | 7 | 11 | 1170.4 | 1121.3 | 12 |
| 55187358 | 08-02 13:53 | 08-06 12:57 | 316 | 188-128 | 9 | 158 | 1148.8 | 1132.2 | 9 |
| 55217189 | 08-03 16:48 | 08-06 12:45 | 140 | 74-66 | 7 | - | 1059.2 | 946.3 | 6 |

**Deck repertoire** (36 replays sampled, 2 distinct exact lists):

- **Marnie's Grimmsnarl ex** — 22 sampled games, subs [55183271, 55187358]
  - 10x Basic {D} Energy (7), 2x Froslass (104), 4x Munkidori (112), 4x Marnie's Impidimp (646), 3x Marnie's Morgrem (647), 3x Marnie's Grimmsnarl ex (648), 2x Snorunt (860), 3x Rare Candy (1079), 1x Unfair Stamp (1080), 4x Buddy-Buddy Poffin (1086), 3x Night Stretcher (1097), 1x Pokégear 3.0 (1122), 1x Tool Scrapper (1137), 4x Poké Pad (1152), 2x Boss’s Orders (1182), 4x Team Rocket's Petrel (1219), 4x Lillie's Determination (1227), 1x Dawn (1231), 4x Spikemuth Gym (1259)
- **Mega Kangaskhan ex** — 14 sampled games, subs [55217189]
  - 1x Basic {G} Energy (1), 4x Mist Energy (11), 4x Spiky Energy (14), 4x Grow Grass Energy (18), 3x Dwebble (344), 3x Crustle (345), 4x Mega Kangaskhan ex (756), 4x Buddy-Buddy Poffin (1086), 4x Crushing Hammer (1120), 4x Pokégear 3.0 (1122), 4x Switch (1123), 4x Jumbo Ice Cream (1147), 1x Hero’s Cape (1159), 2x Boss’s Orders (1182), 4x Xerosic’s Machinations (1197), 4x Hilda (1225), 4x Lillie's Determination (1227), 2x Battle Cage (1264)

No within-submission list variation observed in the sample: one exact 60-card list per submission.

**Matchups (sampled games, W-L by opponent archetype):**

| opponent archetype | W | L |
|---|---|---|
| Marnie's Grimmsnarl ex | 7 | 4 |
| Fezandipiti ex | 5 | 2 |
| Mega Lopunny ex | 3 | 4 |
| Mega Kangaskhan ex | 0 | 4 |
| Dragapult ex | 1 | 1 |
| Hydrapple ex | 1 | 0 |
| Teal Mask Ogerpon ex | 1 | 0 |
| Cynthia's Garchomp ex | 0 | 1 |
| Mega Venusaur ex | 1 | 0 |
| Team Rocket's Mewtwo ex | 0 | 1 |
