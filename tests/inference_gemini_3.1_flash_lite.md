# Inference Results - Gemini 3.1 Flash Lite Preview

## Configuration
- **Model**: `gemini-3.1-flash-lite-preview`
- **Temperature**: `0.1`
- **Max Tokens**: `2048`
- **Task Max Steps**: `Easy: 40, Medium: 100, Hard: 150`
- **Environment**: HR Recruiting OpenEnv (v1.0.0)

## Terminal Output
```text
PS C:\Users\rushi\Music\Cave\Lab\Projects\CHROME> venv\scripts\activate
(venv) PS C:\Users\rushi\Music\Cave\Lab\Projects\CHROME> python -m hr.inference

==================================================
  Task 0 (Easy)
==================================================
  Step 1: hire_candidate → Hired #1 → Engineering @ ₹8.0L
  Step 2: hire_candidate → Hired #5 → Engineering @ ₹8.0L
  Step 3: hire_candidate → Hired #12 → Engineering @ ₹8.0L
  Step 4: hire_candidate → Hired #24 → Support @ ₹5.0L
  Step 5: hire_candidate → Hired #74 → Support @ ₹5.0L
  Step 6: hire_candidate → Hired #77 → Support @ ₹5.0L
  Step 7: hire_candidate → Hired #14 → Marketing @ ₹8.0L
  Step 8: hire_candidate → Hired #26 → Marketing @ ₹8.0L
  Step 9: hire_candidate → Hired #39 → Sales @ ₹8.0L
  Step 10: hire_candidate → Hired #19 → Sales @ ₹5.0L
  Step 11: hire_candidate → Hired #21 → QA @ ₹8.0L
  Step 12: hire_candidate → Hired #51 → QA @ ₹8.0L

  ✅ Complete! Grader: 0.7877

==================================================
  Task 1 (Medium)
==================================================
  Step 1: hire_candidate → Hired #3 → Engineering @ ₹12.0L
  Step 2: hire_candidate → Hired #41 → Engineering @ ₹12.96L
  Step 3: hire_candidate → Hired #14 → Engineering @ ₹13.92L
  Step 4: hire_candidate → Hired #12 → Engineering @ ₹14.88L
  Step 5: hire_candidate → Hired #22 → Engineering @ ₹18.0L
  Step 6: hire_candidate → Hired #95 → DataScience @ ₹15.84L
  Step 7: hire_candidate → Hired #13 → DataScience @ ₹8.0L
  Step 8: hire_candidate → Hired #106 → DataScience @ ₹8.64L
  Step 9: hire_candidate → Hired #108 → DataScience @ ₹16.8L
  Step 10: hire_candidate → Hired #35 → Marketing @ ₹5.0L
  Step 11: hire_candidate → Hired #100 → Marketing @ ₹9.28L
  Step 12: hire_candidate → Hired #17 → Sales @ ₹5.4L
  Step 13: hire_candidate → Hired #33 → Marketing @ ₹5.8L
  Step 14: hire_candidate → Hired #67 → Sales @ ₹6.2L
  Step 15: hire_candidate → Hired #50 → Support @ ₹19.44L
  Step 16: hire_candidate → Hired #101 → Marketing @ ₹6.6L
  Step 17: hire_candidate → Hired #160 → QA @ ₹9.92L
  Step 18: hire_candidate → Hired #282 → DevOps @ ₹20.88L
  Step 19: hire_candidate → Hired #102 → ProductMgmt @ ₹22.32L
  Step 20: hire_candidate → Hired #68 → Sales @ ₹7.0L
  Step 21: hire_candidate → Hired #70 → Sales @ ₹7.4L
  Step 22: hire_candidate → Hired #105 → Support @ ₹10.56L
  Step 23: hire_candidate → Hired #130 → Support @ ₹3.0L
  Step 24: hire_candidate → Hired #10 → QA @ ₹11.2L
  Step 25: hire_candidate → Hired #112 → Support @ ₹11.84L
  Step 26: hire_candidate → Hired #168 → QA @ ₹17.76L
  Step 27: hire_candidate → Hired #127 → DevOps @ ₹18.72L
  Step 28: hire_candidate → Hired #120 → DevOps @ ₹19.68L
  Step 29: hire_candidate → Hired #146 → ProductMgmt @ ₹20.64L
  Step 30: hire_candidate → Hired #196 → ProductMgmt @ ₹21.6L
  Step 31: hire_candidate → Hired #198 → Design @ ₹12.48L
  Step 32: hire_candidate → Hired #19 → Design @ ₹13.12L
  Step 33: hire_candidate → Insufficient budget (₹13.12L left)

  💸 Out of Budget! Stopping early. Grader: 0.606

==================================================
  Task 2 (Hard)
==================================================
  Step 1: hire_candidate → Hired #3 → Engineering @ ₹12.0L
  Step 2: hire_candidate → Hired #141 → AI_Research @ ₹18.54L
  Step 3: hire_candidate → Hired #14 → Engineering @ ₹13.32L
  Step 4: hire_candidate → Hired #41 → Engineering @ ₹13.92L
  Step 5: hire_candidate → Hired #12 → Engineering @ ₹14.88L
  Step 6: hire_candidate → Hired #15 → DataScience @ ₹15.84L
  Step 7: hire_candidate → Hired #36 → Engineering @ ₹16.8L
  Step 8: hire_candidate → Hired #22 → Engineering @ ₹19.98L
  Step 9: hire_candidate → Hired #95 → DataScience @ ₹18.12L
  Step 10: hire_candidate → Hired #97 → DataScience @ ₹18.72L
  Step 11: hire_candidate → Hired #50 → AI_Research @ ₹21.42L
  Step 12: hire_candidate → Hired #25 → DataScience @ ₹20.04L
  Step 13: hire_candidate → Hired #46 → DataScience @ ₹8.24L
  Step 14: hire_candidate → Hired #54 → AI_Research @ ₹21.0L
  Step 15: hire_candidate → Hired #32 → AI_Research @ ₹21.6L
  Step 16: hire_candidate → Hired #2 → DevOps @ ₹8.88L
  Step 17: hire_candidate → Hired #42 → DevOps @ ₹22.92L
  Step 18: hire_candidate → Hired #334 → Security @ ₹22.86L
  Step 19: hire_candidate → Hired #35 → Security @ ₹5.15L
  Step 20: hire_candidate → Hired #49 → DevOps @ ₹23.88L
  Step 21: hire_candidate → Hired #34 → DevOps @ ₹11.44L
  Step 22: hire_candidate → Hired #107 → Security @ ₹11.84L
  Step 23: hire_candidate → Hired #10 → ProductMgmt @ ₹12.48L
  Step 24: hire_candidate → Hired #100 → ProductMgmt @ ₹13.12L
  Step 25: hire_candidate → Hired #19 → ProductMgmt @ ₹13.76L
  Step 26: hire_candidate → Hired #130 → ProductMgmt @ ₹3.09L
  Step 27: hire_candidate → Hired #184 → Marketing @ ₹3.24L
  Step 28: hire_candidate → Hired #17 → Marketing @ ₹5.55L
  Step 29: hire_candidate → Hired #335 → Sales @ ₹3.57L
  Step 30: hire_candidate → Hired #67 → Marketing @ ₹5.95L
  Step 31: hire_candidate → Hired #101 → Marketing @ ₹6.2L
  Step 32: hire_candidate → Hired #395 → Marketing @ ₹3.81L
  Step 33: hire_candidate → Hired #68 → Sales @ ₹6.75L
  Step 34: hire_candidate → Hired #499 → Design @ ₹24.84L
  Step 35: hire_candidate → Hired #481 → Sales @ ₹7.0L
  Step 36: hire_candidate → Hired #497 → QA @ ₹14.64L
  Step 37: hire_candidate → Hired #33 → Sales @ ₹7.55L
  Step 38: hire_candidate → Hired #448 → Support @ ₹24.3L
  Step 39: hire_candidate → Hired #132 → Sales @ ₹7.8L
  Step 40: hire_candidate → Hired #163 → Finance @ ₹15.28L
  Step 41: hire_candidate → Hired #420 → Design @ ₹4.05L
  Step 42: hire_candidate → Hired #160 → Legal @ ₹15.68L
  Step 43: hire_candidate → Hired #498 → HR @ ₹4.2L
  Step 44: hire_candidate → Hired #440 → Design @ ₹16.32L
  Step 45: hire_candidate → Hired #443 → Design @ ₹16.96L
  Step 46: hire_candidate → Hired #131 → QA @ ₹27.72L
  Step 47: hire_candidate → Hired #128 → QA @ ₹17.84L
  Step 48: hire_candidate → Hired #470 → QA @ ₹18.24L
  Step 49: hire_candidate → Hired #139 → Support @ ₹8.35L
  Step 50: hire_candidate → Hired #118 → Support @ ₹8.6L
  Step 51: hire_candidate → Hired #496 → Logistics @ ₹28.68L
  Step 52: hire_candidate → Hired #409 → CustomerSuccess @ ₹19.12L
  Step 53: hire_candidate → Hired #111 → Support @ ₹9.15L
  Step 54: hire_candidate → Insufficient budget (₹4.77L left)

  💸 Out of Budget! Stopping early. Grader: 0.5096

==================================================
  Scores: Easy=0.7877 | Medium=0.6060 | Hard=0.5096
  Average: 0.6344
==================================================
```
