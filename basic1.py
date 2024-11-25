import cv2
import pandas as pd

# Load the color dataset
color_data = pd.read_csv("colors.csv", names=["color", "color_name", "hex", "R", "G", "B"])

# Function to find the closest color name
def get_color_name(R, G, B):
    minimum = 10000
    cname = ""
    for i in range(len(color_data)):
        d = abs(R - int(color_data.loc[i, "R"])) + abs(G - int(color_data.loc[i, "G"])) + abs(B - int(color_data.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = color_data.loc[i, "color_name"]
    return cname

# Mouse click callback function
def draw_function(event, x, y, flags, param):
    global clicked, r, g, b, xpos, ypos
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

# Read the image
image_path = "image.jpg"  # Replace with your image path
img = cv2.imread(image_path)
img = cv2.resize(img, (800, 600))  # Resize for better visibility

# Global variables
clicked = False
r = g = b = xpos = ypos = 0

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_function)

while True:
    cv2.imshow("Image", img)
    if clicked:
        # Display color name and RGB values
        text = get_color_name(r, g, b) + f' R={r} G={g} B={b}'
        cv2.rectangle(img, (20, 20), (600, 60), (b, g, r), -1)
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        clicked = False

    # Break the loop
    if cv2.waitKey(20) & 0xFF == 27:  # Exit on pressing 'ESC'
        break

cv2.destroyAllWindows()
