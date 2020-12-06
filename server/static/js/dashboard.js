(function () {
    let $ctrl = this;

    const launchTraining = (button, trainingType) => {
        button.disabled = true;
        fetch('/training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                'training_type': trainingType
            })
        })
            .then((response) => {
                if (response.status === 200) {
                    console.log('Training started');
                }
            })
            .catch((error) => {
                // There was an error
                console.warn('Error launching the training:', error);
            })
            .finally(() => {
                button.disabled = false;
            });

    }

    const init = () => {
        $ctrl.mnistTrainingButton = document.getElementById('mnistTrainingButton');
        $ctrl.mnistTrainingButton.addEventListener('click', () => {
            launchTraining(this, 'MNIST');
        }, false);

        $ctrl.chestXRayTrainingButton = document.getElementById('chestXRayTrainingButton');
        $ctrl.chestXRayTrainingButton.addEventListener('click', () => {
            launchTraining(this, 'CHEST_X_RAY_PNEUMONIA');
        }, false);
    }

    init();
})();
