# Explaining Trading Via Image Classification

The focus of the paper is on explainability and not profitability (probably).

This is the paper extending / using the results from Gideon thesis.

**Research questions**:

* Given that it is possible to learn complex arithmetic rules (decisions from technical indicators) using a visual representation of the data, can we visually explain the decisions behind these patterns?

* Do I really need to train classifiers? Why not visually explain common technical indicators? In the end the trained classifiers are only trying to reproduce the underlying rules from each technical indicator.

**Methodology**

* Label according to technical indicator signals and obtain images according to the paper Trading Via Image Classification.

* Train classifiers and divide the train dataset in images classified as buy and images classified as no-buy. Let us call this dataset the knowledge base (KB).

* Obtain a vector-based representation of each image in the KB. Use a ResNet18 as explained in Chapter 4 of the thesis.

* Grab an image from the test set (the query image) and obtain the label assigned from a classifier as well as its vector-based representation.

 * Here I have two options:
    * Use the assigned label for the query image and retrieve from the KB the most similar neighbors **that have the same label** as the query image. Then explain using S/D maps. This would explain in what parts of the chart the model focused the most as well as the shape of the charts.

    * Retrieve the most similar neighbors from the KB even if they don't share the same label as the query image. Then explain using S/D maps. Since neighbors can have different labels, probably this approach is not as interpretable as the previous one but I'll just keep it in mind. It might be possible that, for neighbors with different labels, we can interpret the S/D maps as the part that the model does not consider relevant (two images agreeing in certains areas and yet having distintic labels).

* **Possible research** How can we learn an autoenconder that has high visual similarity and at the same time uses the labels?

* Even when the paper *Trading Via Image Classification* is not truly focused on developing trading strategies, using the proposed methodology we could develop a simple strategy.
    * Once we retrieve the neighbors of a query image, we can count how many of these neighbors had a correct label (remember that J.P. Morgan paper uses technical indicators to label the data, therefore we have false positive signals). To decide if a neighbor has a correct label we have to specify a look-ahead time window for the neighbor or see if in the future there is a sell signal. We can even weight the importance of a neighbor using the similarity with respect to the query image. In the end, if there is enough support, then we take the trading decision of the label assigned to the query image (or the weighted majority vote if the labels of the neighbors are mixed).
