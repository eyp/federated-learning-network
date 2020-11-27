(function () {

    let launchTrainingButton = document.getElementById("launchTrainingButton");

    const launchTraining = () => {
        launchTrainingButton.disabled = true;
        fetch('/training', {
            method: 'POST'
        })
            .then((response) => {
                if (response.status === 200) {
                    console.log('Training started');
                }
            })
            .finally(() => {
                launchTrainingButton.disabled = false;
            })

    }

    launchTrainingButton.addEventListener('click', () => {
        launchTraining();
    })

    const init = () => {
    }

    init();
})();
