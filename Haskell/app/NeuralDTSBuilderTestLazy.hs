{-# LANGUAGE OverloadedStrings #-}

module NeuralDTSBuilderTestLazy where

import qualified Data.Map.Strict as M
import qualified Data.Text.Lazy as LazyT
import qualified Data.Text.Lazy.IO as LazyTIO
import qualified Data.Text.Read as TR
import Data.Maybe (mapMaybe)
import Data.List (maximum)
import NeuralDTSBuilderUtils (angleChild, angleParent, sigmoid)

-- word,dim1,dim2,dim3,... 形式でロードする(単語に対して複数の埋め込みを格納)
loadWordEmbeddings :: FilePath -> IO (M.Map LazyT.Text [[Float]])
loadWordEmbeddings path = do
    contents <- LazyTIO.readFile path
    let ls = drop 1 $ LazyT.lines contents

    let parseFloat t =
            case TR.double (LazyT.toStrict t) of
              Right (d, _) -> Just (realToFrac d :: Float)
              _ -> Nothing

    let parseLine line =
            case LazyT.splitOn "," line of
              (wordTxt : dimsTxt) ->
                  let dims = mapMaybe parseFloat dimsTxt
                  in Just (wordTxt, dims)
              _ -> Nothing

    let parsedLines = mapMaybe parseLine ls
    pure $ M.fromListWith (++) [(word, [dims]) | (word, dims) <- parsedLines]


calcPScore :: Float -> [Float] -> [Float] -> Float
calcPScore alpha pEmb cEmb =
    let a1 = angleChild alpha pEmb cEmb
        a2 = angleParent alpha pEmb cEmb
        score = a1 - a2
    in sigmoid score

neuralDTSBuilder :: IO (LazyT.Text -> LazyT.Text -> Float)
neuralDTSBuilder = do
    let embPath   = "data/NeuralDTSBuilderTest.csv"
    let alpha     = 0.1
    let notFoundValue = 0.0 :: Float

    embMap <- loadWordEmbeddings embPath

    let oracle :: LazyT.Text -> LazyT.Text -> Float
        oracle parent child =
            case (M.lookup parent embMap, M.lookup child embMap) of
                (Just pEmbs, Just cEmbs) ->
                    let allScores =
                            [ calcPScore alpha pEmb cEmb
                            | pEmb <- pEmbs
                            , cEmb <- cEmbs
                            ]
                    in if null allScores
                       then notFoundValue
                       else maximum allScores
                _ -> notFoundValue

    pure oracle


main :: IO ()
main = do
    oracle <- neuralDTSBuilder
    print $ oracle "有袋動物" "経済"
    print $ oracle "有袋動物" "カンガルー"
    print $ oracle "有袋動物" "aaa"
    print $ oracle "艦隊" "アルカリ土類金属"