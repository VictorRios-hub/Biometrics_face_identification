import tensorflow as tf

def contrastive_loss(y_pred, y_true, margin=1.0):
    # y_true : étiquette de similarité (0 pour différentes, 1 pour similaires)
    # y_pred : Prédiction de similarité (distance euclidienne)
    # margin : Marge pour la perte (distance seuil entre images similaires et différentes)
    
    # Move the tensors to the CPU to convert them to NumPy arrays
    y_pred_cpu = y_pred.cpu().detach().numpy()
    y_true_cpu = y_true.cpu().detach().numpy()
    
    # Calculer la perte pour les images similaires
    loss_similar = y_true * 0.5 * tf.square(y_pred)

    # Calculer la perte pour les images différentes
    loss_different = (1 - y_true) * 0.5 * tf.square(tf.maximum(margin - y_pred, 0))

    # Perte totale : somme des pertes pour les images similaires et différentes
    total_loss = loss_similar + loss_different

    return total_loss

def total_loss(y_true, y_pred):
    return tf.reduce_sum(y_pred)
