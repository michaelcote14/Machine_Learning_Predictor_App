import sqlite3

def row_deleter():
    # Create a database or connect to one that exists
    conn = sqlite3.connect('Predictions_Database')

    # Create a cursor instance
    cursor = conn.cursor()

    data = ['G3']

    # Delete from database
    cursor.execute(
        'DELETE from predictions_table WHERE Target = :Target',
        {'Target': data[0]})
    # Commit changes
    conn.commit()

    # Close our connection
    conn.close()

def inserter():
    # Create a database or connect to one that exists
    conn = sqlite3.connect('Predictions_Database')

    # Create a cursor instance
    cursor = conn.cursor()

    data = ['12/19/22', 'test_target', 20, 'test_dataframe', 'test_model', 0.333, 0.1, 4.4]

    # Create table in database if it doesn't already exist
    cursor.execute("""CREATE TABLE if not exists predictions_table (
            date_predicted DATE,
            target text,
            predicted_value real,
            dataframe text,
            model_used text,
            normal_score real,
            r2_score real,
            mean_absolute_error real
            )""")

    cursor.execute("INSERT INTO predictions_table VALUES (:date_predicted, :target, :predicted_value, :dataframe, :model_used, :normal_score, :r2_score, :mean_absolute_error)",
                   {'date_predicted': data[0],
                    'target': data[1],
                    'predicted_value': data[2],
                    'dataframe': data[3],
                    'model_used': data[4],
                    'normal_score': data[5],
                    'r2_score': data[6],
                    'mean_absolute_error': data[7]
                   })
    # Commit changes
    conn.commit()

    # Close connection
    conn.close()


def column_adder():
    # Create a database or connect to one that exists
    conn = sqlite3.connect('Training_Model_Database')

    # Create a cursor instance
    cursor = conn.cursor()

    cursor.execute('ALTER TABLE training_model_table ADD COLUMN Total_Model_Upgrades')

    conn.commit()
    conn.close()

def column_deleter():
    # Create a database or connect to one that exists
    conn = sqlite3.connect('Training_Model_Database')

    # Create a cursor instance
    cursor = conn.cursor()

    cursor.execute('ALTER TABLE training_model_table DROP COLUMN Total_Model_Upgrades')

    conn.commit()
    conn.close()

row_deleter()

