In this experiment you will evaluate Self Prompting SD3.5 verison 2.


Self Prompting SD3.5 weights: /home/ekim339/project/SD3.5/CODEX/self_prompting_sd35/checkpoints/version2/checkpoint-030000

Check the files under /home/ekim339/project/SD3.5/CODEX/self_prompting_sd35 to look at self prompting sd3.5 pipeline. Dataprocessing and inputs to network are completely different from TextCtrl.

I want to sample 100 images with 5 character texts from the SRNet_Datagen dataset. Now randomly add noises to each of the samples. You will use these same 100 samples for the following evaluations.

In fact, these 100 images are already sampled in previous experiment. Use the same images and noise levels for this experiment as well.

Check this directory for the samples: /home/ekim339/project/SD3.5/experiments/self_prompting_sd35_version1/results/samples.jsonl

For this experiment I want you to use the exact same strings from the previous experiment as the target text.

check target_text column of this file: /home/ekim339/project/SD3.5/experiments/self_prompting_sd35_version1/results/detailed_results.csv

For each experiments below, you will first check 'target_key' to check the target strings corresponding to that experiment. Then use the exact combinations of 'target_text' and 'filename' in the csv file.


# a) Capital and lowercase target text

1. Use Self Prompting SD3.5 version 2 to edit the text. In the csv file there are randomly sampled 5-character-words from the previous experiment which you will as a target text. 

You will use the target_text of rows where target_key='case_upper' and target_key='case_lower'. The 'case_upper' and 'case_lower' rows are paired with each other and is the exact same string but with uppercase and lower case.  

Using the 100 noise pertubed images discussed previously, edit the source text to target text. 

e.g. row with target_key='case_upper' has target_text='XAQKY' and another row has target_ley='case_lower' and target_text='xaqky'. In the csv, check filename column of these rows. If filename='08945.png' then edit the source text of this image to both 'XAQKY' and 'xaqky'

2. Use OCR detector to detect the generated texts from self prompting SD3.5. Compute the metrics ACC, NED, and CER using the true target text and detected target text. Report the mean and standadard variations of these metrics. Store the results in csv file. 

You will create a copy of this csv file /home/ekim339/project/SD3.5/experiments/self_prompting_sd35_version1/results/capital_lowercase_summary.csv and concatenate on its rows and columns.

In csv, there are two tables, each for capitalized target text and lowercase target text.
- Add a row for 'self promptding sd3.5 version 2' for these two tables. The table has metrics on columns (mean of ACC, std of ACC, mean of NED, std of NED, mean of CER, std of CER). Fill this part in.

3. Now you will create and store a collaged image for visualization.
- There will be 3 columns; original noise added sample, self prompting verison 1 capital, self prompting version 2 lowercase
- There will be 5 rows; use the first 5 samples
- Store the generated/edited images in 3\*5 collage (First column is the source image with noise added)
- Above each images for column 1, write the source text.
- Above each images for column 2 and 3, write the target text. Below the target text, write ACC, NED, and CER of OCR detected texts

layout for col 2 and 3:
{target text label}
ACC: ?, NED: ?, CER: ?
{generated image}

# b) Letters and special characters target text

1. Use Self prompting SD3.5 version 2 to edit the text.

Initial plan is to
  - 1-1) Randomly sample 5 letters and use it as a target text 1. 
  - 1-2) Randomly sample 4 letters and 1 special character and use it as a target text 2. Position/location of the special character does not matter.
    - special character indicates characters such as '.', '!', '?', ',' etcs
  - 1-3) Randomly sample 3 letters and 2 special character and use it as a target text 3. 
  - 1-4) Randomly sample 2 letters and 1 special character and use it as a target text 4. 
  - 1-5) Randomly sample 1 letters and 4 special character and use it as a target text 5. 
  - 1-6) Randomly sample 5 special characters and use it as a target text 6. 

and then edit the soruce text of 100 samples into these 6 target texts.

However, these strings are already sampled in the csv file. check the column target_key.
- 1-1) corresponds to target_key='special_1'
- 1-2) corresponds to target_key='special_2'
- 1-3) corresponds to target_key='special_3'
- 1-4) corresponds to target_key='special_4'
- 1-5) corresponds to target_key='special_5'
- 1-6) corresponds to target_key='special_6'

then use the target_key of the corresponding rows.

2. Use OCR detector to detect the generated texts from self prompting SD3.5 version 2. Compute the metrics ACC, NED, and CER using the true target text and detected target text. Report the mean and standadard variations of these metrics for TextCtrl and self prompting SD3.5. Store the results in csv file.

You will create a copy of this csv file /home/ekim339/project/SD3.5/experiments/self_prompting_sd35_version1/results/special_character_summary.csv and concatenate on its rows and columns.

In csv, there is a single table
- Add a row for self prompting sd3.5 verison 2. Then fill in the metrics for the columns.
  - The columns are: (mean/std of ACC for target text 1, mean/std of NED for target text 1, mean/std of CER for target text 1) repeat this for all 6 target texts. There would be 18 columns in total.
  - each cell of the table would store '{mean}/{std}' of a metric. i.e. show both mean and std of the metric in a single cell.

3. Now you will create and store a collaged image for visualization.
- There will be 7 columns; original noise added sample + one for each target text
- There will be 1 row for the samples
- Store the generated/edited images along this single row
- Above each images for column 1, write the source text.
- Above each images for column 2-7, write target text and ACC, NED, and CER of OCR detected texts

layout for col 2-7:
{target text label}
ACC: ?, NED: ?, CER: ?
{generated image}
