{-# LANGUAGE OverloadedStrings #-}

module NeuralDTSBuilderTest where

import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Read as TR
import Data.Maybe (mapMaybe)
import NeuralDTSBuilderUtils (angleChild, angleParent, sigmoid)

-- word,dim1,dim2,dim3,... 形式でロードする
loadWordEmbeddings :: FilePath -> IO (M.Map String [Float])
loadWordEmbeddings path = do
    contents <- TIO.readFile path
    let ls = drop 1 $ T.lines contents

    let parseFloat t =
            case TR.double t of
              Right (d, _) -> Just (realToFrac d :: Float)
              _ -> Nothing

    let parseLine line =
            case T.splitOn "," line of
              (wordTxt : dimsTxt) ->
                  let dims = mapMaybe parseFloat dimsTxt
                  in Just (T.unpack wordTxt, dims)
              _ -> Nothing

    pure $ M.fromList (mapMaybe parseLine ls)


-- Embedding（word→embedding）でOracleの実装
neuralDTSBuilder :: IO (String -> String -> Float)
neuralDTSBuilder = do
    let embPath   = "data/NeuralDTSBuilderTest.csv"
    let alpha     = 0.1
    let threshold = 0.00
    let notFoundValue = 0.0 :: Float -- 確率なので見つからない場合は0にしておく

    embMap <- loadWordEmbeddings embPath

    -- oracle :: (DTTdB.ConName -> DTTdB.ConName -> Float) にしたい (DTTdB.ConNameの呼び出し方わからない)
    let oracle :: String -> String -> Float
        oracle parent child =
            case (M.lookup parent embMap, M.lookup child embMap) of
                (Just pEmb, Just cEmb) ->
                    let a1 = angleChild alpha pEmb cEmb
                        a2 = angleParent alpha pEmb cEmb
                        score = a1 - a2
                        pScore = sigmoid score
                        pTh    = sigmoid threshold
                    in pScore
                _ -> notFoundValue

    pure oracle


main :: IO ()
main = do
    oracle <- neuralDTSBuilder
    print $ oracle "有袋動物" "経済"
    print $ oracle "有袋動物" "カンガルー"
    print $ oracle "有袋動物" "aaa"
    print $ oracle "艦隊" "アルカリ土類金属"
