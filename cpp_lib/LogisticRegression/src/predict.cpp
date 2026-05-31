/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   predict.cpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: nhanafi <nhanafi@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/05/31 15:42:55 by nhanafi           #+#    #+#             */
/*   Updated: 2026/05/31 16:15:27 by nhanafi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "LinearRegression.hpp"

Vector LinearRegression::predict(const Matrix& X, const Vector& coefficients, double bias) {
    size_t m = X.getRows();
    Vector predictions(m);

    predictions = static_cast<Vector>(X * coefficients) + Vector(m, bias);
    return predictions; 
}


Vector LinearRegression::predict(const Matrix& X) const {
    return LinearRegression::predict(X, _coefficients, _bias);
}

double LinearRegression::predictSingle(const Vector& x) const {
    return LinearRegression::predict(x, _coefficients, _bias)(0);
}

