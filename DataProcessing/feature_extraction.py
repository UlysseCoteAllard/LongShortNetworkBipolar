import numpy as np
import pandas as pd
import neurokit2 as nk


def getRMSSD(vector):
    diff_vector = np.zeros(len(vector) - 1)
    for i in range(len(vector) - 1):
        diff_vector[i] = vector[i + 1] - vector[i]
    intermediary_value = np.mean(np.power(diff_vector, 2))
    rmssd = np.sqrt(intermediary_value)
    return rmssd


def getSampEn(vector, m=2, r=.2):
    # The default r is 0.2 times the standard deviation if dimension = 2
    result = nk.entropy_sample(signal=vector, dimension=m, r=r)
    return result


def getRootSquaredMoment2(vector, myLen):
    squared_vector = np.square(vector)
    return np.sqrt(np.sum(squared_vector, where=(np.isfinite(squared_vector))) / (myLen - 1))


def getRootSquaredMoment(vector):
    squared_vector = np.square(vector)
    return np.sqrt(np.sum(squared_vector, where=(np.isfinite(squared_vector))))


def getSparseness(moment_zero, moment_two, moment_four, epsilon=1e-15):
    denominator = np.sqrt(np.abs((moment_zero - moment_two) * (moment_zero - moment_four))) + epsilon
    sparseness = moment_zero / denominator
    return np.log(sparseness, where=(sparseness > 0))


def getIrregularityFactor(moment_zero, moment_two, moment_four, epsilon=1e-15):
    moment_zero_time_moment_four = moment_zero * moment_four
    denominator = (np.sqrt(moment_zero_time_moment_four, where=(np.isfinite(moment_zero_time_moment_four)))) + epsilon
    irregularity_factor = moment_two / denominator
    return np.log(irregularity_factor, where=(irregularity_factor > 0))


def getStandardDeviationPercentageOfMean(vector):
    std = np.std(vector)
    mean = np.mean(vector)
    return std / mean


def getFeatures_for_tsd_bipolar(vector, epsilon=1e-15):
    vector = np.array(vector)
    lambda_variable = 0.1
    first_derivative = np.diff(vector, axis=0) + epsilon
    second_derivative = np.diff(first_derivative, axis=0) + epsilon
    moment_zero = (np.power(getRootSquaredMoment(vector), lambda_variable) / lambda_variable) + epsilon
    moment_two = (np.power(getRootSquaredMoment2(first_derivative, len(vector)),
                           lambda_variable) / lambda_variable)
    moment_four = (np.power(getRootSquaredMoment2(second_derivative, len(vector)),
                            lambda_variable) / lambda_variable) + epsilon

    moment_zero_minus_moment_two = moment_zero - moment_two
    moment_zero_minus_moment_four = moment_zero - moment_four
    # Sparseness
    sparseness = getSparseness(moment_zero=moment_zero, moment_two=moment_two, moment_four=moment_four)
    # Irregularity factor
    irregularity_factor = getIrregularityFactor(moment_zero=moment_zero, moment_two=moment_two, moment_four=moment_four)
    # Teager-Kaiser energy operator
    before_sum_TEA = first_derivative[0:len(first_derivative) - 1] ** 2 - np.multiply(vector[0:len(vector) - 2],
                                                                                      second_derivative)
    TEA = np.nansum(np.where(np.isclose(before_sum_TEA, np.inf), np.nan, before_sum_TEA))
    Feat = [np.log(moment_zero, where=(moment_zero > 0)),
            np.log(moment_zero_minus_moment_two, where=(moment_zero_minus_moment_two > 0)),
            np.log(moment_zero_minus_moment_four, where=(moment_zero_minus_moment_four > 0)), sparseness,
            irregularity_factor, TEA]
    return Feat


def standard_bipolar(vector):
    features = [getRMSSD(vector=np.abs(vector)) / np.std(np.abs(vector)),
                getStandardDeviationPercentageOfMean(vector=vector),
                getSampEn(vector=vector)]
    return features


def tsd(vector):
    features_vector = getFeatures_for_tsd_bipolar(vector=vector)
    return np.array(features_vector, dtype=np.float32)


def tsd_bipolar(vector):
    features_vector = getFeatures_for_tsd_bipolar(vector=vector)
    features_vector.extend(standard_bipolar(vector=vector))
    return np.array(features_vector, dtype=np.float32)


def eda_feature_extraction(vector, frequency):
    eda_signal = nk.signal_sanitize(vector)
    # This function applies a low pass filter at 3Hz, by doubling the stated sampling rate, the low pass
    # filter is applied at 1.5Hz which is what we want to extract.
    eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=frequency * 2., method="neurokit")
    eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=frequency)
    # Find peaks
    peak_signal, info = nk.eda_peaks(
        eda_decomposed["EDA_Phasic"].values,
        sampling_rate=frequency,
        method="neurokit",
        amplitude_min=0.1,
    )
    # Store
    df = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})
    df = pd.concat([df, eda_decomposed, peak_signal], axis=1)
    analyzed_df = nk.eda_analyze(df, sampling_rate=frequency)

    formatted_example = analyzed_df.iloc[0].values
    formatted_example = np.append(formatted_example, nk.eda_autocor(df, sampling_rate=frequency))
    formatted_example = formatted_example[1:]
    return formatted_example


def ppg_feature_extraction(vector, frequency):
    signals, info = nk.ppg_process(ppg_signal=vector, sampling_rate=frequency)
    hrv_df = nk.hrv(signals["PPG_Peaks"], sampling_rate=frequency, show=False)
    formatted_example = [hrv_df["HRV_SDNN"].values[0], hrv_df["HRV_SampEn"].values[0]]
    return formatted_example


if __name__ == '__main__':
    array_a = [4.0364, 9.3522, -7.4884, -29.285, -6.1338, 3.0227, 22.24, 56.407, 34.754, 40.978, 23.505, 9.8179, 35.561,
               14.979, 17.96, -16.515, -28.289, -40.667, 38.803, 22.503, 19.205, -19.043, -34.123, -24.346, -54.224,
               -28.449,
               -6.1375, -70.389, -36.899, -2.1967, 13.465, -0.89412, 7.9374, 3.5035, 52.504, 36.984, 4.1532, 25.537,
               41.873,
               38.147, 13.645, -38.204, -102.55, -53.543, 33.253, 45.099, 8.0257, 10.686, 22.726, 52.178, 9.3122,
               -19.47,
               -4.2596, 12.7, 19.97, -1.3481, -24.034, -30.646, -23.297, -29.211, -27.832, -27.049, -38.635, -46.151,
               11.309,
               -2.4908, 30.031, 52.965, 28.569, 38.263, 28.971, 17.451, 37.29, 51.586, 51.582, 26.725, -23.095, -13.071,
               -127.68, -72.734, -45.092, -55.192, -28.271, -51.371, -53.138, -31.197, -1.6199, 49.398, 101.34, 68.465,
               58.786, 35.03, 31.726, 43.093, 42.144, 1.6165, -40.013, -75.465, -43.826, 24.753, 72.596, 87.152, 54.129,
               14.602, -44.362, -68.031, -52.01, -77.875, -98.936, -81.045, -27.242, 74.372, 107, 85.314, 37.729,
               1.7476,
               -3.0569, 12.673, -132.35, -74.879, -54.101, 5.1472, 44.755, -10.166, 10.312, 9.2823, 30.675, 41.601,
               8.9585,
               -1.0946, 4.2738, -12.697, -2.2753, 5.8905, 8.3999, 11.748, 24.368, -63.069, -79.326, 12.473, 62.336,
               65.376,
               61.344, 49.618, -2.3384, -13.851, -14.773, -39.104, -30.04, -17.228, -42.235]
    array_b = [-169.14, 35.79, 404.24, 853.98, 1272, 1819.5, 2289.8, 2599.2, 2293.9, 1765.7, 974.72, -822.49, -1378,
               -1028.8,
               -159.75, 1149.3, 1633.2, 1628.8, 1099.9, -1673.1, -1352, -833.11, -391.54, 240.37, 1185.8, 1146.9,
               147.54,
               -1545.7, -1901.9, -1912.7, -1385.4, -799.15, 264.87, 918.46, 1199.1, -72.77, -1169.6, -1469, -1761.6,
               -1464.6,
               -730.7, -117.85, 409.76, 1669.5, 1808.3, 1718.4, 817.19, -1141.8, -1319.3, -1122.8, -23.83, 583.36,
               1207.2,
               1903.1, 2226.9, 1260, 528.09, -1.6473, -549.28, -894.44, -1066.4, -939.12, -765.56, -354.39, 461.14,
               1618.2,
               1812.8, 1521.5, -419.03, -867.18, -1319.1, -1507.7, -1495.5, -1232, -871.99, -328, 773.63, 1479.8,
               551.91,
               -490.93, -896.28, -923.17, -499.76, 45.458, 1177, 1783.6, 2120.8, 1891.3, 35.742, -538.28, -1038.7,
               -1352.5,
               -1450.8, -1290.3, -319.44, 212.48, 789.03, 1250.8, 1391.6, 1005.9, 417.76, -262.59, -852.6, -1347.5,
               -1951.1,
               -1105.3, -538.15, 104.97, 735.39, 1193.3, 1290.7, 821.05, 206.97, -919.64, -1334.6, -1631.7, -1508.3,
               -777.65,
               1073.3, 2148.7, 1917, 774.68, -774.44, -1094.3, -1099, -764.75, -290.68, 673.75, 1293.4, 1744.6, 1902,
               1716.3,
               1419.4, 770.38, 139.77, -479.92, -1797.9, -1769.7, -1374.6, 486.1, 900.99, 854.1, 672.77, 179.75,
               -1123.2,
               -2729.4, -2660.8, -2498.4, -1042.4, -458.06, 925.83]
    array_c = [13.578, 53.829, -65.68, -200.31, -156.64, -139.58, -137.16, -149.81, -198.99, -168.64, -118.72, 5.3937,
               165.24,
               38.995, -94.329, 24.045, 20.555, -21.383, 43.058, 143.96, 71.508, -53.514, -58.626, -60.29, -88.772,
               -10.751,
               3.7171, 13.774, 3.5227, 12.969, 14.325, 18.502, 102.77, 107.98, 13.426, -21.051, 8.7797, 28.82, 76.196,
               64.903,
               47.919, 13.381, -12.934, 22.309, 98.637, 65.633, -117.47, -63.797, 1.5475, 21.797, -152.55, -17.855,
               168.95,
               247.96, 107.76, -141.05, -213.4, -237.12, -154.74, -102.47, -71.443, -40.231, -18.62, -12.484, 17.69,
               36.76,
               73.499, 142.54, 25.863, -25.085, -36.134, 19.989, 127.61, 194.86, 146.18, 23.604, -119.34, -105.69,
               -32.869,
               -32.964, -28.362, -17.177, 0.21596, -10.303, -35.115, -34.329, 81.652, 113.74, 60.512, -105.09, -104.55,
               -22.854, -2.5456, -16.565, 58.494, 83.75, 55.548, -19.934, -53.41, -20.641, -34.402, -49.394, -34.447,
               17.79,
               0.43814, -101.11, -46.762, 4.9029, -4.0583, 2.8058, 88.535, 346.59, 381, 170.23, -174.66, -334.27,
               -133.48,
               -119.91, -200.43, -92.701, -66.501, -6.5912, 45.353, 105.46, 136.19, 124.53, 69.078, -129.32, -67.908,
               9.9804,
               39.303, 49.505, 28.771, 12.308, -30.453, -54.57, 6.6631, 31.894, 36.134, -62.292, -32.043, 5.857, 2.1077,
               -2.6899, 120.55, 84.02, -7.7783, -30.075, -103.94, -91.047, -85.004]
    print("")
    print("SAMPEN: ")
    print(getSampEn(array_a))
    print(getSampEn(array_b))
    print(getSampEn(array_c))
