import tkinter as tk
from customtkinter import CTkEntry, CTkLabel, CTkButton
from PIL import ImageTk

from diffusers import StableDiffusionPipeline
import torch
from torch import autocast


# Function to authenticate (assuming auth_token function exists)
def get_auth_token():
    return auth_token()  # Replace with your authentication logic

# Function to generate image
def generate_image():
    with autocast("cuda"):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
        image.save("generatedimage.png")
        img = ImageTk.PhotoImage(image)
        lmain.configure(image=img)


# Create the main app window
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Entry field for user prompt
prompt = CTkEntry(
    height=40, width=512, text_font=("Arial", 20), text_color="black", fg_color="white"
)
prompt.place(x=10, y=10)

# Label to display the generated image
lmain = CTkLabel(height=512, width=512)
lmain.place(x=10, y=110)

# Model ID and device configuration
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=get_auth_token
)
pipe.to(device)

# Generate button with click event
generate_button = CTkButton(
    height=40,
    width=120,
    text_font=("Arial", 20),
    text_color="white",
    fg_color="blue",
    command=generate_image,
)
generate_button.configure(text="Generate")
generate_button.place(x=206, y=60)

# Run the main application loop
app.mainloop()