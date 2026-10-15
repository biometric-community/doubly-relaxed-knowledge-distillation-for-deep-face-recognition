# DEVIATIONS — DR-KD

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train data | CASIA-WebFace | CelebA | CASIA not local |
| D2 | Teacher | Pretrained FR ResNet50 | ImageNet init + CE on CelebA | No CASIA teacher ckpt |
| D3 | Student zoo | Multiple students / losses | ResNet-18 only | Lab twin scope |
| D4 | Eval | + AgeDB, CFP-FP, IJB | LFW + CALFW + CPLFW | Available locally |
| D5 | Epochs | Paper schedule | Teacher 6 + student 10 | Full-train budget |
