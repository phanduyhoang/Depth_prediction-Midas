import cv2
import torch
import matplotlib.pyplot as plt

# Load MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Load transforms
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Get correct image size
    height, width, _ = img.shape 

    # Run MiDaS depth prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(height, width), 
            mode='bicubic',
            align_corners=False
        ).squeeze()
        output = prediction.cpu().numpy()  

    # Display depth map
    plt.imshow(output, cmap='plasma')
    plt.pause(0.00001)

    # Show original frame
    cv2.imshow('CV2Frame', frame)

    # Quit on 'q' press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.show()
