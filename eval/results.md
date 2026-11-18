# Lost & found search evaluation

Gallery: **171 indexed items** from 9 cameras. Targets: 50 hand-labelled items.

### Detection coverage (target appears in the index)

| Detectors | Targets indexed |
|---|---|
| YOLO-seg only | 16/50 |
| + YOLO-World | 37/50 |
| + static-object detector | 38/50 |

Not indexed: red umbrella, white coffee cup, turquoise backpack, kids green backpack, bag on cafe chair, navy lettered bag, brown paper bag, pink kids backpack, yellow bucket hat, brown striped bag, black sling bag, light blue backpack

### Text search: embedding model, crop and view aggregation (no boosts)

| Setting | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| ViT-B/32, masked crop (old model) | 10% | 28% | 34% | 0.19 |
| ViT-L/14, masked crop | 24% | 32% | 38% | 0.29 |
| ViT-L/14, context crop | 20% | 34% | 46% | 0.28 |
| ViT-L/14, masked + context fused | 22% | 32% | 44% | 0.28 |
| ViT-L/14, fused, best view (max) | 20% | 42% | 48% | 0.29 |
| ViT-L/14, fused, mean + best view | 22% | 42% | 46% | 0.30 |
| ViT-L/14, context crop, best view (app default) | 24% | 42% | 52% | 0.32 |

### Text search (50 descriptions)

| Setting | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| single prompt template | 22% | 46% | 52% | 0.31 |
| + colour boost | 18% | 44% | 46% | 0.28 |
| + category + colour boost | 16% | 40% | 48% | 0.26 |
| app default | 24% | 42% | 52% | 0.32 |

### Photo search (12 photos)

| Setting | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| CLIP | 33% | 50% | 67% | 0.42 |
| DINOv2 | 42% | 58% | 58% | 0.49 |
| CLIP + DINOv2, averaged vectors | 50% | 67% | 75% | 0.57 |
| CLIP + DINOv2 (app default) | 42% | 67% | 67% | 0.53 |

### Text queries (app default)

| Target | Query | Rank |
|---|---|---|
| wall clock | "wall clock" | 1 |
| brown sling bag | "brown leather sling bag" | 51 |
| black backpack | "black backpack" | 4 |
| red umbrella | "red umbrella" | not indexed |
| pink handbag | "pink handbag" | 65 |
| white bag | "white canvas bag" | 81 |
| black handbag | "black handbag" | 6 |
| pink pattern bag | "pink bag with white pattern" | 1 |
| white coffee cup | "white coffee cup" | not indexed |
| patterned cup | "paper coffee cup with colorful pattern" | 1 |
| studded backpack | "black backpack with metal studs" | 5 |
| white tote bag | "white tote bag" | 96 |
| grey handbag | "grey leather handbag" | 5 |
| zebra bag | "black and white zebra print bag" | 1 |
| white shopping bags | "white shopping bags" | 4 |
| black trolley bag | "black shopping bag on a wheeled trolley" | 6 |
| black plastic bag | "black plastic bag" | 1 |
| blue backpack | "dark blue backpack" | 2 |
| yellow shopping bag | "yellow shopping bag" | 1 |
| pink shopping bag | "pink shopping bag" | 64 |
| striped bag | "black and white striped shopping bag" | 1 |
| smartphone | "black smartphone" | 8 |
| turquoise backpack | "turquoise backpack" | not indexed |
| white plastic bag | "white plastic bag" | 10 |
| kids green backpack | "green patterned child's backpack" | not indexed |
| bag on cafe chair | "black bag left on a cafe chair" | not indexed |
| teal bag on chair | "teal bag on a chair" | 5 |
| navy lettered bag | "navy blue bag with white lettering" | not indexed |
| brown paper bag | "brown paper shopping bag" | not indexed |
| yellow backpack | "yellow backpack" | 1 |
| pink kids backpack | "pink child's backpack" | not indexed |
| orange suitcase | "orange wheeled suitcase" | 2 |
| printed white bag | "white bag with black print" | 18 |
| yellow bucket hat | "yellow bucket hat" | not indexed |
| red handbag | "red handbag" | 1 |
| cream handbag | "cream leather handbag" | 2 |
| mustard crossbody bag | "mustard yellow crossbody bag" | 11 |
| blue puffer jacket | "bright blue puffer jacket" | 1 |
| grey quilted jacket | "grey quilted jacket" | 10 |
| red shopping bag | "red shopping bag" | 18 |
| white shopping bag | "white shopping bag" | 25 |
| tan handbag | "small tan leather handbag" | 1 |
| dark blue bag | "dark blue bag" | 34 |
| navy tote bag | "navy blue tote bag" | 38 |
| brown striped bag | "brown striped handbag" | not indexed |
| black sling bag | "black sling bag" | not indexed |
| red backpack | "red backpack" | 1 |
| phone in clear case | "smartphone in a clear case" | 4 |
| black lap handbag | "black handbag" | 38 |
| light blue backpack | "light blue backpack" | not indexed |

### Photo queries (app default)

| Photo of | Rank |
|---|---|
| yellow shopping bag | 1 |
| turquoise backpack | not indexed |
| white plastic bag | 2 |
| brown paper bag | not indexed |
| yellow backpack | 1 |
| orange suitcase | 2 |
| yellow bucket hat | not indexed |
| cream handbag | 1 |
| blue puffer jacket | 1 |
| grey quilted jacket | 16 |
| red shopping bag | 1 |
| tan handbag | 3 |
