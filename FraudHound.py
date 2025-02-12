
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

# Add custom CSS styles
st.markdown(
     """
    <style>
    /* Custom CSS */
    body {
        background-color: #EEE2DC;
        color: #EDC787;
    }
    .css-erpbzb.edgvbvh3{
        visibility: hidden;
    }
    .css-cio0dv.egzxvld1{
        visibility: hidden;
    }
    .css-ab6luk.e19lei0e1{
        visibility: hidden;
    }
    .stButton button {
        background-color: #FF0000;
        color: #FFFFFF;
    }

    /* Add more custom styles as needed */

    </style>
    """,
    unsafe_allow_html=True
)

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)


# Define the different app pages
image_file = "https://i.postimg.cc/5tPsT1SB/imagelog.png"
st.image(image_file)
def page_home():
    with st.container() :
        
        st.title("FraudHound")
        
        st.write("A Machine Learning model for fraud detection")
        image_file22221 = "https://i.postimg.cc/MpdN25RZ/Screenshot-2023-05-14-152814.png"
        st.image(image_file22221,)
    
    with st.container() :
        st.write("---")
        left_column,right_column = st.columns(2)
        st.header("About the Project")
        st.write(""" FraudHound is designed to identify fraudulent transactions in credit card data.
          It utilizes a logistic regression algorithm trained on a balanced dataset consisting of both
            legitimate and fraudulent transactions. The model takes input features related to credit card transactions
              and predicts whether a transaction is legitimate or fraudulent.  The model aims to provide an efficient and
                  reliable tool for detecting and preventing credit card fraud, contributing to enhanced security in financial transactions.""")

   




def page_prediction():
    
    st.header("Make a Prediction")
    st.write("Enter transaction details below:")




    # create input fields for the user to enter feature values
    input_df = st.text_input('')
    input_df_lst = input_df.split(',')
    # create a button to submit input and get prediction
    submit = st.button("Submit")

    if submit:
        # get input feature values
        features = np.array(input_df_lst, dtype=np.float64)
        # make prediction
        prediction = model.predict(features.reshape(1, -1))
        # display result
        if prediction[0] == 0:
            st.header("✅ Legitimate transaction")
        else:
            st.header("❌ Fraudulent transaction")
def page_about():
    st.title("THE OUTCOMES")
    st.header("Reduction in Financial Losses")
    image_file1 = "https://i.postimg.cc/s2NN1S0Q/1.jpg"
    st.image(image_file1,)
    st.write("By detecting and preventing fraudulent transactions, credit card fraud detection systems help minimize financial losses for both individuals and businesses.")
    st.write("---")
    st.header("Enhanced Customer Trust")
    image_file21 = "https://i.postimg.cc/m2s6cTVw/2.jpg"
    st.image(image_file21,)
    st.write(" Implementing effective fraud detection measures fosters customer trust by ensuring their financial security and protecting them from fraudulent activities.")
    st.write("---")
    st.header("Lower Operational Costs")
    image_file231 = "https://i.postimg.cc/VNKFDCyf/image.png"
    st.image(image_file231,)
    st.write(" By automating the detection and prevention of fraudulent activities, credit card fraud detection systems help reduce manual efforts and operational costs associated with investigating and resolving fraud cases.")
    st.write("---")
    st.header("Early Fraud Warning")
    image_file2331 = "https://i.postimg.cc/J7J24s30/4.jpg"
    st.image(image_file2331,)
    st.write(" Fraud detection systems can provide early warnings and alerts for suspicious activities, allowing timely intervention and prevention of potential fraud.")
    st.write("---")


def page_contact():
    st.header("Contact Us")
    st.markdown("""
    <form action="https://formsubmit.co/teamcodecrushers@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="text" name="name" placeholder="your name" required>
    <input type="email" name="email" placeholder="your email" >
    <textarea name="message" placeholder="enter your message">
    </textarea>
    <input type="file" name="attachment" accept="image/png, image/jpeg">
    <button type="submit">Submit</button>
    </form>
    <style>
    input[type=text], select, textarea {
  width: 100%; /* Full width */
  padding: 12px; /* Some padding */ 
  border: 1px solid #ccc; /* Gray border */
  border-radius: 4px; /* Rounded borders */
  box-sizing: border-box; /* Make sure that padding and width stays in place */
  margin-top: 6px; /* Add a top margin */
  margin-bottom: 16px; /* Bottom margin */
  resize: vertical /* Allow the user to vertically resize the textarea (not horizontally) */
}

/* Style the submit button with a specific background color etc */
input[type=submit] {
  background-color: #04AA6D;
  color: white;
  padding: 12px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}



</style>""",  
  
    unsafe_allow_html=True
)
  
  
    

# Create a dictionary of the app pages
pages = {
    "Home": page_home,
    "Test": page_prediction,
    "Outcomes":page_about,
    "Contact": page_contact,
    
}

# Create sidebar navigation

selection = option_menu(
    menu_title=None,
    options=["Home","Test","Outcomes","Contact"],
    icons=["house","credit-card","search","github"],
    default_index=0,
    orientation="horizontal",)

# Run the app
pages[selection]()

st.markdown(
    """  <div>
            <p>Copyright &copy; 2023 by Team Code Crushers- <a href="https://github.com/AbinayReddy2501" target="_parent">Abinay reddy,      </a><a href="https://github.com/Thanmayee7/tan-" target="_parent">Thanmayee-      </a><a href="https://github.com/akshayagajawada" target="_parent">Akshaya-     </a><a href="https://github.com/KandukuriAnish" target="_parent">Anish</a> | all rights reserved.</p>
        </div>""",  
  
    unsafe_allow_html=True
)