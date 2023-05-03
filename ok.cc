/*
 * Copyright (c) 2023-, Pedro Maciel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */


#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>


struct PointLatLon : std::array<double, 3> {
    PointLatLon(double lat, double lon, double value = 0.) : array{lat, lon, value} {}
    double& lat   = array::operator[](0);
    double& lon   = array::operator[](1);
    double& value = array::operator[](2);

    static double distance(const PointLatLon& p, const PointLatLon& q) {
        auto dlon = p.lon - q.lon;
        auto dlat = p.lon - q.lon;
        return std::sqrt(dlat * dlat + dlon * dlon);
    }

    static PointLatLon random() {
        static std::random_device rd{};
        static std::mt19937 gen{rd()};
        static std::normal_distribution<double> d{0., 2};
        return {d(gen), d(gen)};
    };
};


std::ostream& operator<<(std::ostream& out, const PointLatLon& p) {
    return out << '{' << p.lat << ", " << p.lon << '}';
}


struct Variogram {
    Variogram() = default;

    Variogram(const Variogram&)      = delete;
    Variogram(Variogram&&)           = delete;
    void operator=(const Variogram&) = delete;
    void operator=(Variogram&&)      = delete;

    virtual ~Variogram() = default;

    virtual double calculate(double h) = 0;
};


struct Exponential final : Variogram {
    Exponential(double nugget, double sill, double range) : nugget_(nugget), sill_(sill), range_(range) {}

    double calculate(double h) final { return nugget_ + sill_ * (1 - std::exp(-h / range_)); }

    const double nugget_;
    const double sill_;
    const double range_;
};


int main(int argc, char* argv[]) {
    // Ordinary Kriging

    std::vector<PointLatLon> data = {
        {0., 0., 10.}, {1., 3., 20.}, {2., 1., 15.}, {3., 4., 25.}, {4., 2., 30.},
    };

    std::unique_ptr<Variogram> variogram(new Exponential(0., 1., 1.));


    auto n = static_cast<Eigen::Index>(data.size());

    Eigen::MatrixXd A(n + 1, n + 1);
    Eigen::VectorXd b(n + 1);
    Eigen::VectorXd weights;

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            A(i, j) = variogram->calculate(PointLatLon::distance(data[i], data[j]));
        }

        A(i, n) = 1;
        A(n, i) = 1;
    }

    A(n, n) = 0;

    for (size_t c = 0; c < 10; ++c) {
        auto target = PointLatLon::random();

        for (Eigen::Index i = 0; i < n; ++i) {
            b(i) = variogram->calculate(PointLatLon::distance(data[i], target));
        }
        b(n) = 1;

        weights = A.inverse() * b;

        double interpolated = 0.;
        for (int i = 0; i < n; ++i) {
            interpolated += weights(i) * data[i].value;
        }

        std::cout << "Interpolated value at " << target << ": " << interpolated << std::endl;
    }

    return 0;
}
