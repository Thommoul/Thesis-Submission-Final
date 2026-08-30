def plot_store_1_sales():
    import pandas as pd
    import matplotlib.pyplot as plt

    train = pd.read_csv(
        r"C:\Users\30695\Desktop\Programming\python\Files\train.csv"
    )

    train["Date"] = pd.to_datetime(
        train["Date"],
        dayfirst=True
    )

    store_1 = train[train["Store"] == 1].copy()
    store_1 = store_1.sort_values("Date").head(20)

    plt.figure(figsize=(12, 5))

    plt.plot(
        store_1["Date"],
        store_1["Sales"]
    )

    plt.title("Εξέλιξη των ημερήσιων πωλήσεων του καταστήματος 1")
    plt.xlabel("Ημερομηνία")
    plt.ylabel("Ημερήσιες Πωλήσεις")

    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
