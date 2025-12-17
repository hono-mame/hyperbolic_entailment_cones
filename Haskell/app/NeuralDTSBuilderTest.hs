{-# LANGUAGE OverloadedStrings #-}

module NeuralDTSBuilderTest where

import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Read as TR
import Data.Maybe (mapMaybe)
import Data.List (maximum)
import NeuralDTSBuilderUtils (angleChild, angleParent, sigmoid)
import Text.Printf (printf)

-- word,dim1,dim2,dim3,... 形式でロードする(単語に対して複数の埋め込みを格納)
loadWordEmbeddings :: FilePath -> IO (M.Map T.Text [[Float]])
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
                  in Just (wordTxt, dims)
              _ -> Nothing

    let parsedLines = mapMaybe parseLine ls
    pure $ M.fromListWith (++) [(word, [dims]) | (word, dims) <- parsedLines]


calcPScore :: Float -> [Float] -> [Float] -> Float
calcPScore alpha pEmb cEmb =
    let a1 = angleChild alpha pEmb cEmb
        a2 = angleParent alpha pEmb cEmb
        score = a1 - a2
        pScore = sigmoid score
    in pScore


neuralDTSBuilder :: IO (T.Text -> T.Text -> Float)
neuralDTSBuilder = do
    let embPath   = "F1/Embedding_test.csv"
    let alpha     = 0.1
    let threshold = 0.00
    let notFoundValue = 0.0 :: Float

    embMap <- loadWordEmbeddings embPath

    let oracle :: T.Text -> T.Text -> Float
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

printOracle :: (T.Text -> T.Text -> Float) -> T.Text -> T.Text -> IO ()
printOracle oracleFn parent child = do
    let score = oracleFn parent child
    let isEntailment = score > 0.5
    let result = if isEntailment then "包含関係あり (TRUE)" else "包含関係なし (FALSE)"

    putStrLn $ T.unpack parent ++ " > " ++ T.unpack child
    printf "  スコア (PScore): %.4f\n" score
    putStrLn $ "  判定: " ++ result
    putStrLn $ T.unpack (T.replicate 40 "-")

main :: IO ()
main = do
    oracle <- neuralDTSBuilder
    putStrLn "--- Oracle Test ---"
    TIO.putStrLn $ T.replicate 40 "="
    -- Positive examples
    printOracle oracle "車" "スポーツカー"
    printOracle oracle "車" "消防車"
    printOracle oracle "有袋動物" "カンガルー"
    printOracle oracle "食べ物" "マスカルポーネ"
    printOracle oracle "鳥類" "ツバメ"
    printOracle oracle "鳥" "ツバメ"
    printOracle oracle "生き物" "恐竜"
    printOracle oracle "イヌ" "チワワ"
    printOracle oracle "調味料" "タルタルソース"
    printOracle oracle "調味料" "角砂糖"
    printOracle oracle "鳥類" "ペンギン"

    -- Negative examples
    printOracle oracle "車" "カンガルー"
    printOracle oracle "鳥類" "りんご"
    printOracle oracle "食べ物" "オオコウモリ"

