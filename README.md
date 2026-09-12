# Nigerian_House_Price_Predictor

This is my first machine learning project, built right after learning linear and multiple regression

Step 1. Finding the dataset
I found the Nigeria Houses and Prices dataset on Kaggle — 24,326 real house listings scraped from nigeriapropertycentre.com, with bedrooms, bathrooms, toilets, parking space, house type, town, state, and price.
Step 2. Discovering my data was broken
The first thing I did was run df.describe() and i noticed my data and the most expensive house was about ₦1.8 trillion. So i first tried using my IQR to flag outliers, but it wanted to cut about 2,558 rows which was over 10% of my data.
So instead I just researched real Nigerian house prices (Banana Island mansions, and other estates on nigeriaproperty center website ) to find a like right cutoff (~₦25B), and so i only dropped  9 rows that i felt were pretty much the outliers
Step 3. Exploring the data
Step 4. Cleaned up the town column
189 unique towns was too many, so I grouped anything under 50  into "Other" leaving major towns above 50
Step 5. Building the model
I one-hot encoded title, state, and my new grouped town column, then trained a multiple linear regression model. My first attempt was bad  R² of just 0.044. Turns out the skewed price was the problem. Once I log-transformed the price before training, R² jumped to 0.43. Adding the grouped town data on top of that pushed it to 0.65.
Step 6. Checking/Testing it like Honestly
The model does well on typical houses (under ₦2B, ~99.5% of the data) but badly underpredicts ultra-luxury homes, likely due to how few of them exist in the data.
Step 7. Deployment
I saved the model with pickle and built a simple Flask app where you enter house details and get a price prediction.


What I learned
A first ML project isn't about getting a perfect score — it's about understanding why a model behaves the way it does. I learned how to spot and defend outlier decisions, why skew matters, and why location beat room count as a predictor.

Tools I used
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, Flask, Jupyter Notebook
