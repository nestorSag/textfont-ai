from fontai.config.training import Config

class ModelTrainingStage(MLPipelineStage):

  def __init__(self, config: Config):

    self.config = config
    self.data_fetcher = InputPreprocessor()

  def run_from_config(self):

    data_fetcher = InputPreprocessor()
    model = self.config.model 
    model.fit(
      data=data_fetcher.fetch_tfr_files(self.config.input_path.list_files()), 
      steps_per_epoch = args.steps_per_epoch, 
      epochs = args.n_epochs, 
      callbacks=callbacks)

    return model

  def process(self, input_data: Path):

    return self.config.model.predict(input_data)