Kerby and I wanted to find which model (Random Forest, K-Neighbor, Logistic Regression, XGBoost) and which set of basketball
stats lead to the best prediction of the college basketball tournament, March Madness. 

**rough_draft** was our first attempt at just writing down the different steps we would need to do and playing around that
**first_try** was our first model were we tried to place everything into one function
later we decided instead of having one giant function, we would write separate functions for the different steps (cleaning, features, test_train)
and we put all of these functions were placed in the file called **model_functions**
**testing_lists** has the lists of the tournament set up as well as the different lists basketball stats we wanted to test
Everything was combined into **get_predictions**. This file can be run to give you the predicted results of the tournament.
We then decided to add some visualization to show the results in a classic bracket format and this was done in **visualization**
