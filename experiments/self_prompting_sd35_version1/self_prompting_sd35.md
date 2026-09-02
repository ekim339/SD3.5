In this experiment you will compare Self Prompting SD3.5 with TextCtrl + SD1.5.

Self Prompting SD3.5 weights: /home/ekim339/project/SD3.5/CODEX/self_prompting_sd35/checkpoints/checkpoint-050000
TextCtrl weights: /home/ekim339/projects/SD3.5/networks/TextCtrl/weights

Check the files under /home/ekim339/project/SD3.5/CODEX/self_prompting_sd35 to look at self prompting sd3.5 pipeline. Dataprocessing and inputs to network are completely different from TextCtrl.


Sample 100 images with 5 character texts from the SRNet_Datagen dataset. Now randomly add noises to each of the samples. You will use these same 100 samples for the following evaluations.

# a) Capital and lowercase target text

1. Use TextCtrl+SD1.5 to edit the text. Randomly sample 5 characters and use it as a target text. You will generate both capitalized target text and lowercase target text. Using the 100 noise pertubed images discussed previously, edit the source text to target text.

e.g. sampled characters: abcde <br/>
-> then edit the source text to both 'abcde' and 'ABCDE'

2. Repeat the same thing with Self Prompting SD3.5 

3. Use OCR detector to detect the generated texts from TextCtrl and self prompting SD3.5. Compute the metrics ACC, NED, and CER using the true target text and detected target text. Report the mean and standadard variations of these metrics for TextCtrl and self prompting 
SD3.5. Store the results in csv file.

In csv, you will store two tables, each for capitalized target text and lowercase target text.
- each table will have TextCtrl and Self Prompting SD3.5 on rows and metrics on columns (mean of ACC, std of ACC, mean of NED, std of NED, mean of CER, std of CER)

4. Now you will create and store a collaged image for visualization.
- There will be 5 columns; original noise added sample, regular capital, regular lowercase, self prompting capital, self prompting lowercase
- There will be 5 rows; use the first 5 samples
- Store the generated/edited images in 5\*5 collage (First column is the source image with noise added)
- Above each images for column 1, write the source text.
- Above each images for column 2-5, write the target text. Below the target text, write ACC, NED, and CER of OCR detected texts

layout for col 2-5:
{target text label}
ACC: ?, NED: ?, CER: ?
{generated image}

# b) Letters and special characters target text

1. Use TextCtrl+SD1.5 to edit the text.
  - 1-1) Randomly sample 5 letters and use it as a target text 1. 
  - 1-2) Randomly sample 4 letters and 1 special character and use it as a target text 2. Position/location of the special character does not matter.
    - special character indicates characters such as '.', '!', '?', ',' etcs
  - 1-3) Randomly sample 3 letters and 2 special character and use it as a target text 3. 
  - 1-4) Randomly sample 2 letters and 1 special character and use it as a target text 4. 
  - 1-5) Randomly sample 1 letters and 4 special character and use it as a target text 5. 
  - 1-6) Randomly sample 5 special characters and use it as a target text 6. 

Now edit the soruce text of 100 samples into these 6 target texts.

2. Repeat the same thing with self prompting SD3.5.

3. Use OCR detector to detect the generated texts from TextCtrl and self prompting SD3.5. Compute the metrics ACC, NED, and CER using the true target text and detected target text. Report the mean and standadard variations of these metrics for TextCtrl and self prompting SD3.5. Store the results in csv file.

In csv, you will store a single table
- The table will have TextCtrl and Self Prompting SD3.5 on rows and metrics on columns 
  - The columns are: (mean/std of ACC for target text 1, mean/std of NED for target text 1, mean/std of CER for target text 1) repeat this for all 6 target texts. There would be 18 columns in total.
  - each cell of the table would store '{mean}/{std}' of a metric. i.e. show both mean and std of the metric in a single cell.

4. Now you will create and store a collaged image for visualization.
- There will be 7 columns; original noise added sample + one for each target text
- There will be 2 rows; TextCtrl and self prompting SD3.5
- Store the generated/edited images in 2\*7 collage
- Above each images for column 1, write the source text.
- Above each images for column 2-7, write target text and ACC, NED, and CER of OCR detected texts

layout for col 2-7:
{target text label}
ACC: ?, NED: ?, CER: ?
{generated image}

