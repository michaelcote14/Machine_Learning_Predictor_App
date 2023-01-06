import pandas as pd
from Extras.functions import time_formatter
import time

if __name__ == '__main__':
    predictor_data_dict = {'Temperature': [65]}

    from Step_2_Visualizing.visualization import main_visualizer, data_type_cleaner
    main_visualizer(original_df, target_variable)
    type_clean_df = data_type_cleaner(original_df, target_variable)

    from Step_3_Single_Encoding.single_hot_encoder import single_encoder
    single_encoded_df = single_encoder(type_clean_df)

    from Step_4_Multiple_Encoding.multiple_hot_encoder import multiple_encoder
    multiple_encoded_df = multiple_encoder(original_df, single_encoded_df)

    from Step_5_Scaling.scaler import main_scaler
    runtimes = 100
    print('Scaler Predicted Time:', time_formatter(runtimes * 0.9830293601**len(multiple_encoded_df.columns)))
    time.sleep(3)
    scaled_df, scaled_predictor_df = main_scaler(runtimes, multiple_encoded_df, target_variable, predictor_data_dict) #problem line
    print('Scaled Predictor Df:\n', scaled_predictor_df)

    # Run Feature Importance Finder

    # Run Feature Combiner Module

    # Run Trainer Module

    # Run Predictor
    from Step_10_Predicting.predictor import predictor
    predictor(scaled_df, target_variable, scaled_predictor_df)

    # from Step_10_Predicting.predictor import predictor, predictor_plotter
    # all_current_model_predictions, all_pickle_model_predictions = predictor()
    # predictor_plotter(all_current_model_predictions)


    # ToDo create an app that can make this easier and has buttons to click