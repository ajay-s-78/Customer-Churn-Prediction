import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME = os.path.join(BASE_DIR, "customer.db")


def create_database():

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            gender TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            prediction TEXT,
            confidence REAL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(customer_id, gender, tenure,
                    monthly_charges, prediction, confidence):

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (
            customer_id,
            gender,
            tenure,
            monthly_charges,
            prediction,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        customer_id,
        gender,
        tenure,
        monthly_charges,
        prediction,
        confidence
    ))

    conn.commit()

    print("Prediction saved!")
    print("Database:", DATABASE_NAME)
    print("Row ID:", cursor.lastrowid)

    conn.close()


def get_all_predictions():

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY prediction_date DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data


if __name__ == "__main__":
    create_database()
    print("Database created successfully!")