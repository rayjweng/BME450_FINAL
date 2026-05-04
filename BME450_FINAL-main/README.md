# Title

"Grocery and Beverage Image Sorting for Food Pantry Applications."

## Team members

Raymond Weng (rayjweng), Joshua Zhang (ZhangJoshcode)

### Project description

For our Grocery and Beverage Image Sorting project, we are aiming to aid food pantries in sorting commonly donated goods quickly and efficiently. We will use datasets from Kaggle and Google Images to train our neural network to sort items into 9 categories: canned goods, fresh vegetables and fruits, dairy products, meats, shelf-stable proteins (nuts and beans), grains, snacks, beverages, and other. These are categories commonly used in different food pantries throughout the US. Food pantries currently use manual labor to sort donated items for proper storage. Our project would begin automating sorting in this industry and could reduce the costs of food pantries. Our project can also be used to help food pantries keep track of inventory and properly track the expiration of currently stored goods.

Our model will use a convolutional neural network to classify images of commonly donated grocery items, with the use of learning architectures such as ResNet or MobileNet. Input images will be preprocessed through resizing, normalization, and data augmentation techniques to improve model robustness and generalization. We will experiment using learning rate, batch size, number of epochs, optimizer choice, and augmentation stratgies as paramaters to analyze how these impact our model's ouput performance. It will be trained on labeled datasets of grocery and beverage images and evaluated based on accuracy and loss. By comparing architectures and parameter changes, we aim to find the most effective approach for an accurate classification of food pantry items. 

Key challenges include distinguishing visually similar items, handling damaged packaging, and managing diverse product appearances. These will be addressed through dataset expansion, augmentation techniques, and model tuning.

Overall, this project aims to reduce sorting time, lower operational costs, and improve food distribution efficiency within food pantries.
