#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <ctime>
#include <tuple>

using namespace std;

pair<int, vector<pair<int, int>>> hungarian_algorithm(vector<vector<int>>& cost_matrix) {
    int n = cost_matrix.size();
    int m = cost_matrix[0].size();
    vector<vector<int>> origin_cost_matrix = cost_matrix;

    // 步骤 1：行操作 - 每行减去该行的最小值
    for (int i = 0; i < n; ++i) {
        int min_value = *min_element(cost_matrix[i].begin(), cost_matrix[i].end());
        for (int j = 0; j < m; ++j) {
            cost_matrix[i][j] -= min_value;
        }
    }

    // 步骤 2：列操作 - 每列减去该列的最小值
    for (int j = 0; j < m; ++j) {
        int min_value = numeric_limits<int>::max();
        for (int i = 0; i < n; ++i) {
            min_value = min(min_value, cost_matrix[i][j]);
        }
        for (int i = 0; i < n; ++i) {
            cost_matrix[i][j] -= min_value;
        }
    }

    vector<vector<bool>> star_matrix(n, vector<bool>(m, false));
    vector<vector<bool>> prime_matrix(n, vector<bool>(m, false));
    vector<bool> row_covered(n, false);
    vector<bool> col_covered(m, false);

    // 步骤 3：标记每一行中的第一个零，并标记为星号
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (cost_matrix[i][j] == 0 && !row_covered[i] && !col_covered[j]) {
                star_matrix[i][j] = true;
                row_covered[i] = true;
                col_covered[j] = true;
            }
        }
    }

    // 重置覆盖行和列的标记
    fill(row_covered.begin(), row_covered.end(), false);
    fill(col_covered.begin(), col_covered.end(), false);

    // 步骤 4：覆盖每一列中有星号的列
    auto cover_columns_with_stars = [&]() {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (star_matrix[i][j]) {
                    col_covered[j] = true;
                }
            }
        }
    };
    cover_columns_with_stars();

    auto find_zero = [&]() {
        for (int i = 0; i < n; ++i) {
            if (!row_covered[i]) {
                for (int j = 0; j < m; ++j) {
                    if (cost_matrix[i][j] == 0 && !col_covered[j]) {
                        return make_pair(i, j);
                    }
                }
            }
        }
        return make_pair(-1, -1);
    };

    auto find_star_in_row = [&](int row) {
        for (int j = 0; j < m; ++j) {
            if (star_matrix[row][j]) {
                return j;
            }
        }
        return -1;
    };

    auto find_prime_in_row = [&](int row) {
        for (int j = 0; j < m; ++j) {
            if (prime_matrix[row][j]) {
                return j;
            }
        }
        return -1;
    };

    auto augment_path = [&](vector<pair<int, int>>& path) {
        for (const auto& p : path) {
            star_matrix[p.first][p.second] = !star_matrix[p.first][p.second];
        }
    };

    auto adjust_cost_matrix = [&]() {
        int min_uncovered_value = numeric_limits<int>::max();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (!row_covered[i] && !col_covered[j]) {
                    min_uncovered_value = min(min_uncovered_value, cost_matrix[i][j]);
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (row_covered[i]) {
                for (int j = 0; j < m; ++j) {
                    cost_matrix[i][j] += min_uncovered_value;
                }
            }
        }
        for (int j = 0; j < m; ++j) {
            if (!col_covered[j]) {
                for (int i = 0; i < n; ++i) {
                    cost_matrix[i][j] -= min_uncovered_value;
                }
            }
        }
    };

    while (count(col_covered.begin(), col_covered.end(), true) < min(n, m)) {
        auto [row, col] = find_zero();
        if (row == -1) { // 没有未覆盖的零，调整代价矩阵
            adjust_cost_matrix();
            tie(row, col) = find_zero();
        }

        prime_matrix[row][col] = true;
        int star_col = find_star_in_row(row);
        if (star_col == -1) {
            // 增广路径
            vector<pair<int, int>> path;
            path.emplace_back(row, col);
            while (true) {
                int star_row = -1;
                for (int r = 0; r < n; ++r) {
                    if (star_matrix[r][path.back().second]) {
                        star_row = r;
                        break;
                    }
                }
                if (star_row == -1) break;
                path.emplace_back(star_row, path.back().second);

                int prime_col = find_prime_in_row(path.back().first);
                path.emplace_back(path.back().first, prime_col);
            }

            augment_path(path);
            fill(prime_matrix.begin(), prime_matrix.end(), vector<bool>(m, false));
            fill(row_covered.begin(), row_covered.end(), false);
            fill(col_covered.begin(), col_covered.end(), false);
            cover_columns_with_stars();
        } else {
            row_covered[row] = true;
            col_covered[star_col] = false;
        }
    }

    // 计算总代价并返回匹配
    int total_cost = 0;
    vector<pair<int, int>> result;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (star_matrix[i][j]) {
                result.emplace_back(i, j);
                total_cost += origin_cost_matrix[i][j];
            }
        }
    }
    return {total_cost, result};
}

int main() {
    // 输入的 n x m 代价矩阵
    vector<vector<int>> cost_matrix = {
        {12, 7, 9, 7, 9},
        {8, 9, 6, 6, 6},
        {7, 17, 12, 14, 9},
        {15, 14, 6, 6, 10},
        {4, 10, 7, 10, 8}
    };

    // 生成随机的非方阵代价矩阵
    int n = 30, m = 20;
    cost_matrix.clear();
    for (int i = 0; i < n; ++i) {
        vector<int> row(m);
        generate_n(row.begin(), m, []() { return rand() % 100 + 1; });
        cost_matrix.push_back(row);
    }

    clock_t start_time = clock();
    auto [total_cost, matching] = hungarian_algorithm(cost_matrix);
    clock_t end_time = clock();

    cout << "运行时间: " << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << " 秒" << endl;
    cout << "最小总代价: " << total_cost << endl;
    cout << "匹配关系: " << endl;
    for (const auto& match : matching) {
        cout << "任务 " << match.first << " 指派给工人 " << match.second << endl;
    }

    return 0;
}
