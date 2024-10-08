#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <tuple>
#include <ctime>

using namespace std;

class KalmanFilter {
public:
    KalmanFilter() {
        ndim = 4;
        dt = 1.0;
        // 初始化运动矩阵
        motion_mat.resize(2 * ndim, vector<double>(2 * ndim, 0.0));
        for (int i = 0; i < ndim; ++i) {
            motion_mat[i][ndim + i] = dt;
            motion_mat[i + ndim][i + ndim] = 1.0;  // 保持速度部分
            motion_mat[i][i] = 1.0;  // 保持位置部分
        }

        // 初始化更新矩阵
        update_mat.resize(ndim, vector<double>(2 * ndim, 0.0));
        for (int i = 0; i < ndim; ++i) {
            update_mat[i][i] = 1.0;
        }

        std_weight_position = 1.0 / 20;
        std_weight_velocity = 1.0 / 160;
    }

    pair<vector<double>, vector<vector<double>>> initiate(const vector<double>& measurement) {
        vector<double> mean_pos(measurement.begin(), measurement.begin() + 4);
        vector<double> mean_vel(4, 0.0);
        vector<double> mean = mean_pos;
        mean.insert(mean.end(), mean_vel.begin(), mean_vel.end());

        vector<double> std_dev = {
            2 * std_weight_position * measurement[3],
            2 * std_weight_position * measurement[3],
            1e-2,
            2 * std_weight_position * measurement[3],
            10 * std_weight_velocity * measurement[3],
            10 * std_weight_velocity * measurement[3],
            1e-5,
            10 * std_weight_velocity * measurement[3]
        };

        vector<vector<double>> covariance(2 * ndim, vector<double>(2 * ndim, 0.0));
        for (int i = 0; i < 2 * ndim; ++i) {
            covariance[i][i] = std_dev[i] * std_dev[i];
        }
        return {mean, covariance};
    }

    pair<vector<double>, vector<vector<double>>> predict(const vector<double>& mean, const vector<vector<double>>& covariance) {
        vector<double> std_pos = {
            std_weight_position * mean[3],
            std_weight_position * mean[3],
            1e-2,
            std_weight_position * mean[3]
        };
        vector<double> std_vel = {
            std_weight_velocity * mean[3],
            std_weight_velocity * mean[3],
            1e-5,
            std_weight_velocity * mean[3]
        };

        vector<vector<double>> motion_cov(2 * ndim, vector<double>(2 * ndim, 0.0));
        for (int i = 0; i < ndim; ++i) {
            motion_cov[i][i] = std_pos[i] * std_pos[i];
            motion_cov[ndim + i][ndim + i] = std_vel[i] * std_vel[i];
        }

        vector<double> new_mean = mat_vec_mult(motion_mat, mean);
        vector<vector<double>> new_covariance = mat_mult(motion_mat, covariance);
        new_covariance = mat_mult(new_covariance, mat_transpose(motion_mat));
        new_covariance = mat_add(new_covariance, motion_cov);

        return {new_mean, new_covariance};
    }

    pair<vector<double>, vector<vector<double>>> project(const vector<double>& mean, const vector<vector<double>>& covariance) {
        vector<double> std_dev = {
            std_weight_position * mean[3],
            std_weight_position * mean[3],
            1e-1,
            std_weight_position * mean[3]
        };

        vector<vector<double>> innovation_cov(ndim, vector<double>(ndim, 0.0));
        for (int i = 0; i < ndim; ++i) {
            innovation_cov[i][i] = std_dev[i] * std_dev[i];
        }

        vector<double> projected_mean = mat_vec_mult(update_mat, mean);
        vector<vector<double>> projected_cov = mat_mult(update_mat, covariance);
        projected_cov = mat_mult(projected_cov, mat_transpose(update_mat));
        projected_cov = mat_add(projected_cov, innovation_cov);

        return {projected_mean, projected_cov};
    }

    pair<vector<double>, vector<vector<double>>> update(const vector<double>& mean,
                                                        const vector<vector<double>>& covariance,
                                                        const vector<double>& measurement) {
        auto [projected_mean, projected_cov] = project(mean, covariance);

        // 计算Kalman增益
        vector<vector<double>> kalman_gain = mat_mult(covariance, mat_transpose(update_mat));
    
        kalman_gain = mat_mult(kalman_gain, mat_inverse(projected_cov));
        vector<double> innovation = vec_sub(measurement, projected_mean);
        vector<double> new_mean = vec_add(mean, mat_vec_mult(kalman_gain, innovation));
        vector<vector<double>> new_covariance = mat_mult(kalman_gain, projected_cov);
        new_covariance = mat_mult(new_covariance, mat_transpose(kalman_gain));
        new_covariance = mat_sub(covariance, new_covariance);
        return {new_mean, new_covariance};
    }

private:
    int ndim;  // 状态维度
    double dt;  // 时间步长
    double std_weight_position;  // 位置标准权重
    double std_weight_velocity;  // 速度标准权重
    vector<vector<double>> motion_mat;  // 运动矩阵
    vector<vector<double>> update_mat;  // 更新矩阵

    vector<double> mat_vec_mult(const vector<vector<double>>& mat, const vector<double>& vec) {
        vector<double> result(mat.size(), 0.0);
        for (size_t i = 0; i < mat.size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += mat[i][j] * vec[j];
            }
        }
        return result;
    }

    vector<vector<double>> mat_mult(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
        assert(mat1[0].size() == mat2.size());
        vector<vector<double>> result(mat1.size(), vector<double>(mat2[0].size(), 0.0));
        for (size_t i = 0; i < mat1.size(); ++i) {
            for (size_t j = 0; j < mat2[0].size(); ++j) {
                for (size_t k = 0; k < mat2.size(); ++k) {
                    result[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }
        return result;
    }

    vector<vector<double>> mat_add(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
        vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0.0));
        for (size_t i = 0; i < mat1.size(); ++i) {
            for (size_t j = 0; j < mat1[0].size(); ++j) {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        return result;
    }

    vector<vector<double>> mat_sub(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
        vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0.0));
        for (size_t i = 0; i < mat1.size(); ++i) {
            for (size_t j = 0; j < mat1[0].size(); ++j)
            {
                result[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
        return result;
    }

    vector<double> vec_sub(const vector<double>& vec1, const vector<double>& vec2) {
        assert(vec1.size() == vec2.size());
        vector<double> result(vec1.size(), 0.0);
        for (size_t i = 0; i < vec1.size(); ++i) {
            result[i] = vec1[i] - vec2[i];
        }
        return result;
    }

    vector<double> vec_add(const vector<double>& vec1, const vector<double>& vec2) {
        assert(vec1.size() == vec2.size());
        vector<double> result(vec1.size(), 0.0);
        for (size_t i = 0; i < vec1.size(); ++i) {
            result[i] = vec1[i] + vec2[i];
        }
        return result;
    }

    vector<vector<double>> mat_transpose(const vector<vector<double>>& mat) {
        vector<vector<double>> transposed(mat[0].size(), vector<double>(mat.size(), 0.0));
        for (size_t i = 0; i < mat.size(); ++i) {
            for (size_t j = 0; j < mat[0].size(); ++j) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }

    vector<vector<double>> mat_inverse(const vector<vector<double>>& mat) {
        int n = mat.size();
        vector<vector<double>> inverse(n, vector<double>(n, 0.0));
        vector<vector<double>> augmented(n, vector<double>(2 * n, 0.0));

        // 创建增广矩阵
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented[i][j] = mat[i][j];
            }
            augmented[i][i + n] = 1.0;  // 右侧为单位矩阵
        }

        // 高斯消元法
        for (int i = 0; i < n; ++i) {
            // 确保对角线元素不为零
            assert(augmented[i][i] != 0);
            double diag = augmented[i][i];
            for (int j = 0; j < 2 * n; ++j) {
                augmented[i][j] /= diag;  // 归一化当前行
            }

            // 消去其他行的当前列
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    double factor = augmented[j][i];
                    for (int k = 0; k < 2 * n; ++k) {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }

        // 提取逆矩阵
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inverse[i][j] = augmented[i][j + n];
            }
        }

        return inverse;
    }
};

void test_kalman_filter() {
        vector<vector<double>> measurements = {
            {100, 200, 1.5, 50},
            {105, 205, 1.5, 50},
            {110, 210, 1.5, 50},
            {115, 215, 1.5, 50},
            {120, 220, 1.5, 50}
        };
        KalmanFilter kf;
        clock_t start_time = clock();
        auto [mean, covariance] = kf.initiate(measurements[0]);
        clock_t end_time = clock();
        cout << "Time used for initiate: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << " seconds." << endl;

        vector<vector<double>> predictions;

        for (const auto& measurement : measurements) {
            start_time = clock();
            auto [predicted_mean, predicted_cov] = kf.predict(mean, covariance);
            end_time = clock();
            cout << "Time used for predict: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << " seconds." << endl;
            predictions.push_back({predicted_mean[0], predicted_mean[1], predicted_mean[2], predicted_mean[3]});
            start_time = clock();
            auto [updated_mean, updated_covariance] = kf.update(predicted_mean, predicted_cov, measurement);
            end_time = clock();
            cout << "Time used for update: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << " seconds." << endl;
            mean = updated_mean;
            covariance = updated_covariance;
        }

        cout << "Final state mean after updates:" << endl;
        for (const auto& value : mean) {
            cout << fixed << setprecision(4) << value << " ";
        }
        cout << endl;

        cout << "\nPredicted states:" << endl;
        for (const auto& pred : predictions) {
            for (const auto& value : pred) {
                cout << fixed << setprecision(4) << value << " ";
            }
            cout << endl;
        }
    }

int main() {
    test_kalman_filter();
    return 0;
}
