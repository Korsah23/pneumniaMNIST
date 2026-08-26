# Run this at the end of your notebook, after training, to save the weights
torch.save(model.state_dict(), "pneumonia_model.pth")
print("Saved pneumonia_model.pth")
