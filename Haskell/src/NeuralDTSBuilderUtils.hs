{-# LANGUAGE OverloadedStrings #-}

module NeuralDTSBuilderUtils
    ( angleChild
    , angleParent
    , sigmoid
    ) where

import Data.List (zipWith)

eps :: Float
eps = 1e-6

norm :: [Float] -> Float
norm xs = sqrt $ sum $ map (\x -> x*x) xs

dot :: [Float] -> [Float] -> Float
dot xs ys = sum $ zipWith (*) xs ys

clip :: Float -> Float
clip x = max (-1 + eps) (min (1 - eps) x)

sigmoid :: Float -> Float
sigmoid x = 1 / (1 + exp x)

angleChild :: Float -> [Float] -> [Float] -> Float
angleChild k parent child =
    let
        normParent = norm parent
        normParentSq = normParent * normParent
        normChild = norm child
        normChildSq = normChild * normChild
        euclidDist = max (norm $ zipWith (-) parent child) eps
        dotProd = dot parent child
        g = 1 + normParentSq * normChildSq - 2 * dotProd
        gSqrt = sqrt g
        childNumerator = dotProd * (1 + normParentSq) - normParentSq * (1 + normChildSq)
        childDenominator = euclidDist * normParent * gSqrt
        cosAngleChild = clip (childNumerator / childDenominator)
    in acos cosAngleChild

angleParent :: Float -> [Float] -> [Float] -> Float
angleParent k parent _child =
    let
        normParent = norm parent
    in asin (k * (1 - normParent * normParent) / normParent)
