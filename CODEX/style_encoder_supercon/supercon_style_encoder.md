I want to fine-tune a pretrained TextCtrl style encoder using supervised contrastive (SupCon) loss on its glyph/style head.

The TextCtrl style encoder has two output heads:

1. Glyph/style head
2. Spatial head

For this experiment, apply SupCon loss only to the features produced by the glyph/style head. Do not apply the contrastive loss to the spatial head.

Use the SRNet_Datagen dataset for this training process: /home/ekim339/projects/SD3.5/datasets/SRNet_Datagen

### Training procedure

1. Load the pretrained TextCtrl style encoder checkpoint.

2. Freeze the shared encoder/backbone and the spatial head. Only the glyph/style head should be trainable.

3. Construct each minibatch so that it contains:

   * Multiple different styles.
   * At least two images belonging to each style.
   * Preferably different text strings for images sharing the same style.
   * When possible, include the same text string rendered in different styles as hard negatives.

4. For every image \(I_i\) in the minibatch, extract the glyph/style-head representation:

$$
z_i = H_{\text{style}}(E(I_i)).
$$

5. L2-normalize the representations before computing the contrastive loss:

$$
\tilde z_i = \frac{z_i}{\|z_i\|_2}.
$$

6. Apply supervised contrastive loss using the style identity as the class label.

For an anchor image \(i\):

* All other images in the batch with the same style are positive samples.
* Images with different styles are negative samples.

Therefore, SupCon should encourage:

$$
\operatorname{sim}(z_i,z_j)\uparrow
\quad\text{if}\quad
style_i = style_j
$$

and

$$
\operatorname{sim}(z_i,z_j)\downarrow
\quad\text{if}\quad
style_i \neq style_j.
$$

The purpose of this fine-tuning is to make the glyph/style-head representation primarily encode text style while becoming invariant to the actual glyph/text content.

For example:

* `HELLO` in Arial and `WORLD` in Arial should be positives and have similar representations.
* `HELLO` in Arial and `HELLO` in Times should be negatives and have dissimilar representations.

Use the standard supervised contrastive loss over the entire minibatch rather than constructing individual positive/negative pairs manually.

Do not modify the spatial-head output or use it when computing the SupCon loss.