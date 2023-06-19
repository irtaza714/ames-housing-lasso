import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            outliers = ['MS_SubClass', 'Lot_Frontage', 'Lot_Area', 'Overall_Qual',
                        'Overall_Cond', 'Year_Built', 'Mas_Vnr_Area', 'BsmtFin_SF_One',
                        'BsmtFin_SF_Two', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF',
                        'Second_Flr_SF', 'Low_Qual_Fin_SF', 'Gr_Liv_Area', 'Bsmt_Full_Bath',
                        'Bsmt_Half_Bath', 'Full_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr',
                        'TotRms_AbvGrd', 'Fireplaces', 'Garage_Yr_Blt', 'Garage_Cars',
                        'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF', 'Enclosed_Porch',
                        'Ssn_Porch', 'Screen_Porch', 'Pool_Area', 'Misc_Val']
            
            no_outliers_num = ['Year_Remod', 'Half_Bath', 'Mo_Sold', 'Yr_Sold']

            cat = ['MS_Zoning', 'Street', 'Lot_Shape', 'Land_Contour',
                   'Lot_Config', 'Land_Slope', 'Neighborhood', 'Conition_One',
                   'Condition_Two', 'Bldg_Type', 'House_Style', 'Roof_Style', 'Roof_Matl',
                   'Exterior_First', 'Exterior_Second', 'Mas_Vnr_Type', 'Exter_Qual',
                   'Exter_Cond', 'Foundation', 'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure',
                   'BsmtFin_Type_One', 'BsmtFin_Type_Two', 'Heating', 'Heating_QC',
                   'Central_Air', 'Electrical', 'Kitchen_Qual', 'Functional',
                   'Fireplace_Qu', 'Garage_Type', 'Garage_Finish', 'Garage_Qual',
                   'Garage_Cond', 'Paved_Drive', 'Sale_Type', 'Sale_Condition']
            
            outliers_pipeline= Pipeline( steps=
                                        [("imputer",SimpleImputer(missing_values = np.nan, strategy="median")),
                                         ("rs", RobustScaler())] )
            
            no_outliers_num_pipeline = Pipeline( steps=
                                        [("imputer",SimpleImputer(missing_values = np.nan, strategy="mean")),
                                         ("ss", StandardScaler())] )

            cat_pipeline = Pipeline( steps=
                                  [ ('imputer', SimpleImputer(missing_values = np.nan, strategy='most_frequent')),
                                   ('ohe', OneHotEncoder())
                                   ])
            
            preprocessor = ColumnTransformer(
                [
                    ("outliers_pipeline", outliers_pipeline, outliers),
                    ("no_outliers_num_pipeline", no_outliers_num_pipeline, no_outliers_num),
                    ("cat_pipeline", cat_pipeline, cat)
                ]
            )


            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train = pd.read_csv(train_path)

            logging.info("Read train data")
            
            test = pd.read_csv(test_path)

            logging.info("Read test data")

            x_train_transf = train.drop('SalePrice',axis=1)

            logging.info("Dropped target column from the train set to make the input data frame for model training")

            # constant_columns_x_train_transf = x_train_transf.columns[x_train_transf.nunique() == 1]

            # if len(constant_columns_x_train_transf) > 0:
            #      print("Constant columns found in x_train_transf:", constant_columns_x_train_transf)
            # else:
            #     print("No constant columns found in x_train_transf.")

            # logging.info("Checked for constant columns in x_train_transf")

            y_train_transf = train['SalePrice']

            logging.info("Target feature obtained for model training")

            x_test_transf = test.drop('SalePrice', axis=1)

            logging.info("Dropped target column from the test set to make the input data frame for model testing")

            # constant_columns_x_test_transf = x_test_transf.columns[x_test_transf.nunique() == 1]

            # if len(constant_columns_x_test_transf) > 0:
            #      print("Constant columns found in x_test_transf:", constant_columns_x_test_transf)
            # else:
            #     print("No constant columns found in x_test_transf.")

            # logging.info("Checked for constant columns in x_test_transf")      
        
            y_test_transf = test['SalePrice']

            logging.info("Target feature obtained for model testing")

            preprocessor = self.get_data_transformer_object()
            
            logging.info("Preprocessing object obtained")

            x_train_transf_preprocessed = preprocessor.fit_transform(x_train_transf)

            logging.info("Preprocessor applied on x_train_transf")

            x_train_transf_preprocessed_df = pd.DataFrame(x_train_transf_preprocessed)

            logging.info("x_train_transf dataframe formed for backwards elimination")

            for i in range(len(x_train_transf_preprocessed_df.columns)):
                
                x_train_transf_preprocessed_df = x_train_transf_preprocessed_df.rename(columns={x_train_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info("x_train_transf dataframe columns renamed")

            # print ("x_train_transf_preprocessed_df shape before vif:", x_train_transf_preprocessed_df.shape)

            # print ("x_train_transf_preprocessed_df columns before be:", x_train_transf_preprocessed_df.columns)

            # constant_columns_x_train_transf_preprocessed_df = x_train_transf_preprocessed_df.columns[x_train_transf_preprocessed_df.nunique() == 1]

            # if len(constant_columns_x_train_transf_preprocessed_df) > 0:
            #      print("Constant columns found in x_train_transf_preprocessed_df:", constant_columns_x_train_transf_preprocessed_df)
            # else:
            #     print("No constant columns found in x_train_transf_preprocessed_df.")

            # Step 3: Fit the Lasso regression model

            # alphas = [0.0001, 0.001, 0.01, 0.1]
            # alphas = [0.1]

            # for i in alphas:
            #     lasso = Lasso(alpha=i, max_iter=1000000)  # You can adjust the alpha value
            #     lasso.fit(x_train_transf_preprocessed_df, y_train_transf)
                
            #     selected_features = np.where(lasso.coef_ != 0)[0]

            #     print("For alpha =", i)
            #     print(f"Number of selected features: {len(selected_features)}")
            #     print("Selected Features:", selected_features)
            #     # for feature in selected_features:
            #     #     print(feature)
            #     #     print()

            x_train_transf_preprocessed_df_lasso = x_train_transf_preprocessed_df.iloc [:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 204, 206, 207, 209, 210, 211, 212, 213, 214, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228]]

            logging.info("Features selected after through lasso regualarization from x_train_transf_preprocessed_df")

            x_test_transf_preprocessed = preprocessor.transform(x_test_transf)

            logging.info("Preprocessor applied on x_test_transf")

            x_test_transf_preprocessed_df = pd.DataFrame(x_test_transf_preprocessed)

            logging.info('''x_test_transf dataframe formed for slecting the columns already selcted through
                          lasso regularization''')

            for i in range(len(x_test_transf_preprocessed_df.columns)):
                
                x_test_transf_preprocessed_df = x_test_transf_preprocessed_df.rename(columns={x_test_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info("x_test_transf dataframe columns renamed")

            # constant_columns_x_test_transf_preprocessed_df = x_test_transf_preprocessed_df.columns[x_test_transf_preprocessed_df.nunique() == 1]

            # if len(constant_columns_x_test_transf_preprocessed_df) > 0:
            #      print("Constant columns found in x_test_transf_preprocessed_df:", constant_columns_x_test_transf_preprocessed_df)
            # else:
            #     print("No constant columns found in x_test_transf_preprocessed_df.")
            
            x_test_transf_preprocessed_df_lasso = x_test_transf_preprocessed_df.iloc [:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 204, 206, 207, 209, 210, 211, 212, 213, 214, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228]]

            logging.info("features selected from x_test_transf")

            print ("x_train columns after lasso regularization:", x_test_transf_preprocessed_df_lasso.columns)

            # print ("x_train_transf_cfs shape after dropping correlated features::", x_train_transf_preprocessed_df_cfs.shape)

            print ("x_test_transf_be columns after after lasso regularization:", x_test_transf_preprocessed_df_lasso.columns)

            # print ("x_test_transf_be shape after dropping correlated features::", x_test_transf_preprocessed_df_cfs.shape)

            train_arr = np.c_[np.array(x_train_transf_preprocessed_df_lasso), np.array(y_train_transf)]
            
            logging.info("Combined the input features and target feature of the train set as an array.")
            
            test_arr = np.c_[np.array(x_test_transf_preprocessed_df_lasso), np.array(y_test_transf)]
            
            logging.info("Combined the input features and target feature of the test set as an array.")
            
            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor)
            
            logging.info("Saved preprocessing object.")
            
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,)
        
        except Exception as e:
            raise CustomException(e, sys)