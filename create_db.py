import sqlite3

def create_database():
    conn = sqlite3.connect("mental_health.db")
    cursor = conn.cursor()

    with open("init_mental_health.sql", "r", encoding="utf-8") as sql_file:
        sql_script = sql_file.read()

    cursor.executescript(sql_script)
    conn.commit()

    print("База данных mental_health.db успешно создана с таблицами users, emotions и chats.")

    conn.close()

if __name__ == "__main__":
    create_database()
