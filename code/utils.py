from matplotlib import pyplot as plt

def plot_model_probabilities(model, pos_inputs, neg_inputs):
    fig = plt.figure()
    plt.hist(model.predict(pos_inputs),label='positive samples', color='orange', alpha=0.5, bins=100)
    plt.hist(model.predict(neg_inputs),label='negative samples', color='blue', alpha=0.5, bins=100)
    fig.suptitle('thresholds', fontsize=20)
    plt.legend()
    plt.show()