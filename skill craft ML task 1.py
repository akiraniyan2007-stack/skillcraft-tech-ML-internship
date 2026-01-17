from sklearn.linear_model import LinearRegression
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Sample dataset
data = {
    'SquareFootage': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Bathrooms': [1, 2, 2, 3, 3],
    'Price': [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

# Train model
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# GUI
root = tk.Tk()
root.title("House Price Predictor")
root.geometry("700x600")
root.configure(bg="#f0f0f0")

big_font = ("Helvetica", 20)

title_label = tk.Label(
    root,
    text="House Price Predictor",
    font=("Helvetica", 20, "bold"),
    bg="#f0f0f0",
    fg="#2c3e50"
)
title_label.pack(pady=10)

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack()

tk.Label(
    frame,
    text="Square Footage:",
    font=big_font,
    bg="#f0f0f0"
).grid(row=0, column=0, padx=10, pady=5, sticky="e")

sqft_entry = tk.Entry(frame, font=big_font, width=20)
sqft_entry.grid(row=0, column=1)

tk.Label(
    frame,
    text="Bedrooms:",
    font=big_font,
    bg="#f0f0f0"
).grid(row=1, column=0, padx=10, pady=5, sticky="e")

bed_entry = tk.Entry(frame, font=big_font, width=20)
bed_entry.grid(row=1, column=1)

tk.Label(
    frame,
    text="Bathrooms:",
    font=big_font,
    bg="#f0f0f0"
).grid(row=2, column=0, padx=10, pady=5, sticky="e")

bath_entry = tk.Entry(frame, font=big_font, width=20)
bath_entry.grid(row=2, column=1)

result_label = tk.Label(
    root,
    text="",
    font=("Helvetica", 16),
    bg="#f0f0f0",
    fg="green"
)
result_label.pack(pady=10)

fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

def predict_price():
    try:
        sqft = int(sqft_entry.get())
        bed = int(bed_entry.get())
        bath = int(bath_entry.get())

        prediction = model.predict([[sqft, bed, bath]])[0]
        result_label.config(
            text=f"Predicted Price: ${int(prediction)}",
            fg="green"
        )

        ax.clear()
        ax.scatter(df['SquareFootage'], df['Price'], color='blue', label='Original Data')
        ax.scatter(sqft, prediction, color='red', label='Prediction')
        ax.set_title("Square Footage vs Price")
        ax.set_xlabel("Square Footage")
        ax.set_ylabel("Price")
        ax.legend()
        canvas.draw()

    except ValueError:
        result_label.config(text="Please enter valid numbers!", fg="red")

def clear_all():
    sqft_entry.delete(0, tk.END)
    bed_entry.delete(0, tk.END)
    bath_entry.delete(0, tk.END)
    result_label.config(text="")
    ax.clear()
    canvas.draw()

btn_frame = tk.Frame(root, bg="#f0f0f0")
btn_frame.pack(pady=10)

predict_btn = tk.Button(
    btn_frame,
    text="Predict",
    command=predict_price,
    bg="#3498db",
    fg="white",
    padx=10,
    pady=5,
    font=big_font
)
predict_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(
    btn_frame,
    text="Clear",
    command=clear_all,
    bg="#e74c3c",
    fg="white",
    padx=10,
    pady=5,
    font=big_font
)
clear_btn.grid(row=0, column=1, padx=10)

root.mainloop()
