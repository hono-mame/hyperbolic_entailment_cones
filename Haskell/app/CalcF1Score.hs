{-# LANGUAGE OverloadedStrings #-}

module CalcF1Score where

import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Read as TR
import Data.Maybe (mapMaybe)
import Data.List (maximum)
import Text.Printf (printf)

import NeuralDTSBuilderUtils
    ( angleChild
    , angleParent
    , sigmoid
    )

-- word,dim1,dim2,... 形式
loadWordEmbeddings :: FilePath -> IO (M.Map T.Text [[Float]])
loadWordEmbeddings path = do
    contents <- TIO.readFile path
    let ls = drop 1 $ T.lines contents

    let parseFloat t =
            case TR.double t of
              Right (d, _) -> Just (realToFrac d)
              _            -> Nothing

    let parseLine line =
            case T.splitOn "," line of
              (w:ds) ->
                  let dims = mapMaybe parseFloat ds
                  in Just (w, dims)
              _ -> Nothing

    let rows = mapMaybe parseLine ls
    pure $ M.fromListWith (++) [(w, [v]) | (w, v) <- rows]

calcPScore :: Float -> [Float] -> [Float] -> Float
calcPScore alpha pEmb cEmb =
    let a1 = angleChild alpha pEmb cEmb
        a2 = angleParent alpha pEmb cEmb
    in sigmoid (a1 - a2)

buildOracle :: FilePath -> IO (T.Text -> T.Text -> Float)
buildOracle embPath = do
    embMap <- loadWordEmbeddings embPath
    let alpha = 0.1
    let notFoundValue = 0.0

    let oracle parent child =
            case (M.lookup parent embMap, M.lookup child embMap) of
                (Just ps, Just cs) ->
                    let scores =
                            [ calcPScore alpha p c
                            | p <- ps
                            , c <- cs
                            ]
                    in if null scores then notFoundValue else maximum scores
                _ -> notFoundValue

    pure oracle

loadPairs :: FilePath -> IO [(T.Text, T.Text)]
loadPairs path = do
    ls <- TIO.readFile path
    pure
      [ (a,b)
      | line <- T.lines ls
      , let ws = T.splitOn "\t" line
      , length ws == 2
      , let a = ws !! 0
      , let b = ws !! 1
      ]

data Count = Count
    { tp :: Int
    , fp :: Int
    , fn :: Int
    }

emptyCount :: Count
emptyCount = Count 0 0 0

updateCount :: Bool -> Bool -> Count -> Count
updateCount gold pred c
    | gold && pred       = c { tp = tp c + 1 }
    | not gold && pred   = c { fp = fp c + 1 }
    | gold && not pred   = c { fn = fn c + 1 }
    | otherwise          = c

precision :: Count -> Float
precision c =
    let d = tp c + fp c
    in if d == 0 then 0 else fromIntegral (tp c) / fromIntegral d

recall :: Count -> Float
recall c =
    let d = tp c + fn c
    in if d == 0 then 0 else fromIntegral (tp c) / fromIntegral d

f1 :: Count -> Float
f1 c =
    let p = precision c
        r = recall c
    in if p + r == 0 then 0 else 2 * p * r / (p + r)

main :: IO ()
main = do
    let embPath = "F1/dim10_90_hypCones.csv"
    let posPath = "F1/jn_noun_closure.tsv.test"
    let negPath = "F1/jn_noun_closure.tsv.test_neg"
    let threshold_value = 0.061938
    let threshold = sigmoid threshold_value

    oracle <- buildOracle embPath
    posPairs <- loadPairs posPath
    negPairs <- loadPairs negPath

    let evalPair gold (p,c) cnt =
            let score = oracle p c
                pred  = score > threshold
            in updateCount gold pred cnt

    let cnt1 = foldr (evalPair True)  emptyCount posPairs
    let cnt2 = foldr (evalPair False) cnt1       negPairs

    putStrLn "=== Oracle Evaluation ==="
    printf "TP: %d  FP: %d  FN: %d\n" (tp cnt2) (fp cnt2) (fn cnt2)
    printf "Precision: %.4f\n" (precision cnt2)
    printf "Recall:    %.4f\n" (recall cnt2)
    printf "F1 score:  %.4f\n" (f1 cnt2)
