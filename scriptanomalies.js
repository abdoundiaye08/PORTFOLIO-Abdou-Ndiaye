// Gestion des animations au défilement
document.addEventListener("DOMContentLoaded", function () {
    const elements = document.querySelectorAll(".animate-on-scroll");

    function checkScroll() {
        const windowHeight = window.innerHeight;

        elements.forEach(el => {
            const position = el.getBoundingClientRect().top;
            if (position < windowHeight - 50) {
                el.classList.add("visible");
            }
        });
    }

    window.addEventListener("scroll", checkScroll);
    checkScroll();
});

// Gestion du bouton "Accéder au Notebook"
document.getElementById('notebook-button').addEventListener('click', function() {
    const notebookContent = document.getElementById('notebook-content');
    notebookContent.classList.toggle('hidden');

    if (!notebookContent.classList.contains('hidden')) {
        // Étape 1 : Préparation des données
        document.getElementById('code-block-1').textContent = `
            import numpy as np
            import pandas as pd
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.model_selection import train_test_split
            import matplotlib.pyplot as plt

            # Chargement des données
            df = pd.read_csv('sensor_data_with_anomalies.csv')

            # Afficher un échantillon des données
            print(df.head())

            # Normalisation des données entre 0 et 1
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)

            # Vérification de la forme des données normalisées
            print(scaled_data.shape)

            # Fonction pour créer des séquences avec une fenêtre
            def create_sequences(data, sequence_length):
                sequences = []
                labels = []
                for i in range(len(data) - sequence_length):
                    seq = data[i:i+sequence_length]
                    label = data[i+sequence_length]  # Label est la valeur suivante
                    sequences.append(seq)
                    labels.append(label)
                return np.array(sequences), np.array(labels)

            # Définir la longueur de la séquence 
            sequence_length = 10
            X, y = create_sequences(scaled_data, sequence_length)

            # Diviser en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Vérification des formes après transformation
            print(X_train.shape, y_train.shape)
        `;

        // Étape 2 : Création du modèle LSTM
        document.getElementById('code-block-2').textContent = `
            # Création du modèle LSTM
            model = Sequential()

            # Première couche LSTM
            model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))

            # Deuxième couche LSTM
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))

            # Couche de sortie dense (une sortie par valeur prédite)
            model.add(Dense(units=X_train.shape[2]))

            # Compilation du modèle
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Résumé du modèle
            model.summary()
        `;

        // Étape 3 : Entraînement du modèle
        document.getElementById('code-block-3').textContent = `
            # Entraînement du modèle
            history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

            # Visualisation de la courbe de perte
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            plt.show()
        `;

        // Étape 4 : Évaluation des performances du modèle
        document.getElementById('code-block-4').textContent = `
            # Faire des prédictions sur l'ensemble de test
            y_pred = model.predict(X_test)

            # Inverser la normalisation pour interpréter les résultats
            y_test_rescaled = scaler.inverse_transform(y_test)
            y_pred_rescaled = scaler.inverse_transform(y_pred)

            # Calculer la différence (erreur) entre les vraies valeurs et les valeurs prédites
            errors = np.mean(np.abs(y_test_rescaled - y_pred_rescaled), axis=1)

            # Afficher les erreurs pour détecter les anomalies
            print(errors[:10])
        `;

        // Étape 5 : Détection des anomalies
        document.getElementById('code-block-5').textContent = `
            # Définir un seuil d'erreur (par exemple, tout ce qui est au-dessus de 95e percentile est une anomalie)
            threshold = np.percentile(errors, 95)

            # Marquer comme anomalies les prédictions qui dépassent le seuil
            anomalies = errors > threshold

            # Afficher le pourcentage d'anomalies
            print(f"Nombre d'anomalies détectées: {np.sum(anomalies)} sur {len(errors)}")

            # Sauvegarder le modèle
            model.save('lstm_anomaly_model.h5')

            # Sauvegarder le scaler
            import joblib
            joblib.dump(scaler, 'scaler.pkl')
        `;
    }
});