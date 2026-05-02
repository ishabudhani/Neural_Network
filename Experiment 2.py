import pandas as pd

student_dict = {'Name': ['Joe', 'Nat'], 'Age': [20, 21], 'Marks': [85.10, 77.80]}
print(student_dict)

student_df = pd.DataFrame(student_dict)
print(student_df)

df = pd.DataFrame(
    {'A': [1, 2, 3],
     'B': [True, True, False],
     'C': [0.496714, -0.138264, 0.647689]},
    index=['a', 'b', 'c']
)
print(df)

fruits_list = ['Apple', 'Banana', 'Orange', 'Mango']
print(fruits_list)

fruits_df = pd.DataFrame(fruits_list, columns=['Fruits'])
print(fruits_df)

fruits_list = [['Apple', 'Banana', 'Orange', 'Mango'],
               [120, 40, 80, 500]]
print(fruits_list)

fruits_df = pd.DataFrame(fruits_list)
print(fruits_df)

fruits_list = [['Apple', 'Banana', 'Orange', 'Mango'],
               [120, 40, 80, 500]]
print(fruits_list)

fruits_df = pd.DataFrame(fruits_list).transpose()
print(fruits_df)

fruits_list = ['Apple', 'Banana', 'Orange', 'Mango']
price_list = [120, 40, 80, 500]

fruits_df = pd.DataFrame(list(zip(fruits_list, price_list)),
                         columns=['Name', 'Price'])
print(fruits_df)
