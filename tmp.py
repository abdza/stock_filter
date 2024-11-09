import pandas as pd

# Read the Excel file
df = pd.read_excel("your_file.xlsx")

# Define the sessions
sessions = ["Sessi 1", "Sessi 2", "Sessi 3"]


# Function to count 'Hadir' in a column
def count_hadir(column):
    return (column == "Hadir").sum()


# Group by 'kawasan' and count 'Hadir' for each session
result = df.groupby("Kawasan")[sessions].agg(count_hadir)

# Display the result
print(result)
