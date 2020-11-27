(function () {
    let $ctrl = this;

    const launchTraining = () => {
        $ctrl.launchTrainingButton.disabled = true;
        fetch('/training', {
            method: 'POST'
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
                $ctrl.launchTrainingButton.disabled = false;
            })

    }

    const setUpButtons = () => {
        $ctrl.launchTrainingButton.addEventListener('click', () => {
            launchTraining();
        })
    }

    const init = () => {
        $ctrl.launchTrainingButton = document.getElementById("launchTrainingButton");
        setUpButtons();
    }

    init();
})();
