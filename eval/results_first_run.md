# Lost & found search evaluation

Gallery: **293 indexed items** from 14 cameras. Targets: 21 hand-labelled items.

### Detection coverage (target appears in the index)

| Detectors | Targets indexed |
|---|---|
| YOLO-seg only | 3/21 |
| + YOLO-World | 16/21 |
| + static-object detector | 16/21 |

Not indexed: black backpack, red umbrella, pink handbag, white coffee cup, patterned cup

### Text search (21 descriptions)

| Setting | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP, plain crop | 33% | 43% | 52% | 0.40 |
| CLIP, masked crop | 29% | 43% | 52% | 0.37 |
| + category boost | 24% | 48% | 62% | 0.35 |
| + colour boost (app default) | 29% | 52% | 62% | 0.36 |

### Photo search (12 photos)

| Setting | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 58% | 67% | 92% | 0.65 |
| DINOv2 | 75% | 92% | 100% | 0.82 |
| CLIP + DINOv2 (app default) | 75% | 100% | 100% | 0.85 |

### Text queries (app default)

| Target | Query | Rank |
|---|---|---|
| v1 backpack | "black backpack with orange trim" | 10 |
| v2 backpack | "black backpack" | 17 |
| v3 small bag | "small dark blue handbag" | 4 |
| v4 shoulder bag | "black shoulder bag" | 7 |
| v5 plastic bag | "white plastic bag" | 5 |
| v6 small bag | "small silver bag" | 1 |
| v7 purple bag | "purple shopping bag" | 1 |
| v8 suitcase | "silver suitcase" | 1 |
| v9 purple bag | "purple shopping bag" | 1 |
| v10 duffel bag | "black duffel bag" | 1 |
| wall clock | "wall clock" | 1 |
| brown sling bag | "brown leather sling bag" | 65 |
| black backpack | "black backpack" | not indexed |
| red umbrella | "red umbrella" | not indexed |
| pink handbag | "pink handbag" | not indexed |
| white bag | "white canvas bag" | 77 |
| black handbag | "black handbag" | 4 |
| pink pattern bag | "pink bag with white pattern" | 4 |
| white coffee cup | "white coffee cup" | not indexed |
| patterned cup | "paper coffee cup with colorful pattern" | not indexed |
| studded backpack | "black backpack with metal studs" | 3 |

### Photo queries (app default)

| Photo of | Rank |
|---|---|
| v1 backpack | 1 |
| v2 backpack | 1 |
| v3 small bag | 1 |
| v4 shoulder bag | 1 |
| v5 plastic bag | 1 |
| v6 small bag | 2 |
| v7 purple bag | 1 |
| v8 suitcase | 1 |
| v9 purple bag | 1 |
| v10 duffel bag | 1 |
| v9 purple bag -> v7 purple bag | 3 |
| v7 purple bag -> v9 purple bag | 3 |
