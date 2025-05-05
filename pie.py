import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the PostgreSQL database
conn = psycopg2.connect(dbname="testDB", user="postgres", password="Password", host="localhost", port="5432")
cursor = conn.cursor()

# Retrieve data from the context column
cursor.execute("SELECT context FROM HRTABLE10")
rows = cursor.fetchall()

# Convert data into a pandas DataFrame
df = pd.DataFrame(rows, columns=['context'])

# Close cursor and connection
cursor.close()
conn.close()

# Perform any necessary data preprocessing (if needed)

# Calculate the frequency of each context
context_counts = df['context'].value_counts()

# Plot the pie chart
plt.figure(figsize=(10, 6))
plt.pie(context_counts, labels=context_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Contexts')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()
