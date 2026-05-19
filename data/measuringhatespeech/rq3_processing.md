The global disagreement dataset is created by grouping all annotations per `comment_id`. For each comment, the script aggregates annotator labels, computes Shannon entropy and variation ratio, and stores the original text together with annotation statistics for later RQ1 and RQ2 experiments.

After computing disagreement scores, comments are divided into low, medium, and high disagreement bins using the 33rd and 66th percentile entropy thresholds. 

For the persona analysis, annotations are separated into conservative, liberal, and neutral ideological groups using the annotator ideology metadata provided in the dataset. Disagreement metrics are then computed independently within each subgroup.

The persona dataset groups annotations by both `comment_id` and ideological subgroup, producing subgroup-specific disagreement scores for the same comment. This enables later comparison between human ideological disagreement and persona-conditioned LLM outputs for RQ3.

Finally, the processed datasets are exported as CSV files containing the comment text, disagreement bins, entropy scores, annotator IDs, and aggregated labels. Additional balanced sampled files are also generated for direct use in downstream prompting experiments.