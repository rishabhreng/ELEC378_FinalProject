#import the prediction file with the proabilities for each class

def ad_hoc_totally_valid_analysis(prob_boost, prediction_file_probability, clustered_data=False):
    new_pred= prediction_file_probability
    if clustered_data:
        new_pred= prediction_file_probability*(1-2*prob_boost)
        for index in range(len(prediction_file_probability)):
            neighbor_below= max(prediction_file_probability[index-1,;])
            neighbor_above= max(prediction_file_probability[index+1,;])
            for species in prediction_file_probability.columns:
                if species==neighbor_below:
                    new_pred[species]= new_pred[species]+prob_boost
                if species==neighbor_above:
                    new_pred[species]= new_pred[species]+prob_boost
    return new_pred