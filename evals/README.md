# evals

four test cases for the skill itself, in the folder layout that `claude plugin eval` reads. each has a user prompt, input files and a few checks written as plain sentences in `graders/`.

| case | what it checks |
| --- | --- |
| `airline-monthly-baseline-first` | the user asks for numbers only. does the reply still show the baseline comparison, the metric and how many origins were used |
| `leaky-traffic-column` | a column that is measured on the same day as the target. is it kept out, and is its too-good score not sold as accuracy |
| `walmart-turkish-day-first-dates` | a turkish prompt, day-first dates and a weekly season of 52 |
| `random-walk-no-edge` | a series where nothing beats the last value. does the reply say so instead of selling a tiny gap |

run each case with and without the skill and compare how many checks pass. two things keep the comparison fair:

- give the with-skill run a copy of the skill that leaves out this folder. otherwise the agent can list `evals/` and read the case names and the checks.
- grade the message the user receives, not only the files in the output folder. a run can compute the numbers and leave them out of its reply.

the inputs are slices of two public datasets (air passengers and walmart store sales, both cc0) and two synthetic series made for these cases. the prices in `price_daily.csv` are a log walk, so they stay positive.

the cases are not part of ci, because each run is a model call.
