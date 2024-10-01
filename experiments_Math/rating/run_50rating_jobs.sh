
#!/bin/bash

# Loop over rule indices and submit a job for each one
for rule_idx_cmd in {0..49}
do
  python rating_with_50rules.py $rule_idx_cmd
done
