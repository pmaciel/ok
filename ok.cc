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


struct PointLatLon : std::array<double, 2> {
    PointLatLon(double lat, double lon) : array{lat, lon} {}
    double& lat   = array::operator[](0);
    double& lon   = array::operator[](1);

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
    Variogram(double nugget, double sill, double range, double a) : n_(nugget), s_(sill), r_(range), a_(a) {}

    Variogram(const Variogram&)      = delete;
    Variogram(Variogram&&)           = delete;
    void operator=(const Variogram&) = delete;
    void operator=(Variogram&&)      = delete;

    virtual ~Variogram() = default;

    virtual double calculate(double h) = 0;

    const double n_;
    const double s_;
    const double r_;
    const double a_;
};


struct Exponential final : Variogram {
    Exponential(double nugget, double sill, double range, double a) : Variogram(nugget, sill, range, a) {}

    double calculate(double h) final { return (s_ - n_) * (1 - std::exp(-h / (r_ * a_))) + n_; }
};


struct Spherical final : Variogram {
    Spherical(double nugget, double sill, double range, double a) :
        Variogram(nugget, sill, range, a), c1_(3. / (2. * r_)), c2_(0.5 / (r_ * r_ * r_)) {}

    double calculate(double h) final { return (s_ - n_) * (h * c1_ - h * h * h * c2_) + n_; }

    const double c1_;
    const double c2_;
};


struct Gaussian final : Variogram {
    Gaussian(double nugget, double sill, double range, double a) :
        Variogram(nugget, sill, range, a), c_(1. / r_ * r_ * a_) {}

    double calculate(double h) final { return (s_ - n_) * (1. - std::exp(-h * h * c_)) + n_; }

    const double c_;
};


int main(int argc, char* argv[]) {
    // Ordinary Kriging

    std::vector<PointLatLon> points = {{0., 0.}, {1., 3.}, {2., 1.}, {3., 4.}, {4., 2.}};
    std::vector<double> values{10., 20., 15., 25., 30.};

    std::unique_ptr<Variogram> variogram(new Exponential(0., 1., 1., 1.));


    auto n = static_cast<Eigen::Index>(points.size());

    Eigen::MatrixXd M(n + 1, n + 1);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = i + 1; j < n; ++j) {
            M(i, j) = M(j, i) = variogram->calculate(PointLatLon::distance(points[i], points[j]));
        }

        M(i, n) = M(n, i) = 1;
    }

    const auto W = M.inverse();

    for (size_t c = 0; c < 10; ++c) {
        auto target = PointLatLon::random();

        Eigen::VectorXd b(n + 1);
        for (Eigen::Index i = 0; i < n; ++i) {
            b(i) = variogram->calculate(PointLatLon::distance(points[i], target));
        }
        b(n) = 1;

        auto weights = W * b;

        double interpolated = 0.;
        for (int i = 0; i < n; ++i) {
            interpolated += weights(i) * values[i];
        }

        std::cout << "Interpolated value at " << target << ": " << interpolated << std::endl;
    }

    return 0;
}
